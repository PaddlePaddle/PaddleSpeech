#include "cls/nnet/cls_nnet.h"
#ifdef PRINT_TIME
#include <time.h>
#endif

namespace ppspeech {

ClsNnet::ClsNnet(){
    // wav_reader_ = NULL;
    runtime_ = NULL;
};

void ClsNnet::reset(){
    // wav_reader_->Clear();
    ss_.str("");
};

int ClsNnet::init(ClsNnetConf& conf){
    conf_ = conf;
    //init fbank opts
    fbank_opts_.frame_opts.samp_freq = conf.samp_freq;
    fbank_opts_.frame_opts.frame_length_ms = conf.frame_length_ms;
    fbank_opts_.frame_opts.frame_shift_ms = conf.frame_shift_ms;
    fbank_opts_.mel_opts.num_bins = conf.num_bins;
    fbank_opts_.mel_opts.low_freq = conf.low_freq;
    fbank_opts_.mel_opts.high_freq = conf.high_freq;
    fbank_opts_.frame_opts.dither = conf.dither;
    fbank_opts_.use_log_fbank = false;

    //init dict
    if (conf.dict_file_path_ != ""){
        init_dict(conf.dict_file_path_);
    }

    // init model
    fastdeploy::RuntimeOption runtime_option;

#ifdef USE_ORT_BACKEND
    runtime_option.SetModelPath(conf.model_file_path_, "", fastdeploy::ModelFormat::ONNX); // onnx
    runtime_option.UseOrtBackend(); // onnx
#endif
#ifdef USE_PADDLE_LITE_BACKEND
    runtime_option.SetModelPath(conf.model_file_path_, conf.param_file_path_, fastdeploy::ModelFormat::PADDLE);
    runtime_option.UseLiteBackend();
#endif
#ifdef USE_PADDLE_INFERENCE_BACKEND
    runtime_option.SetModelPath(conf.model_file_path_, conf.param_file_path_, fastdeploy::ModelFormat::PADDLE);
    runtime_option.UsePaddleInferBackend();
#endif
    runtime_option.SetCpuThreadNum(conf.num_cpu_thread_);
    runtime_option.DeletePaddleBackendPass("simplify_with_basic_ops_pass");
    runtime_ = std::unique_ptr<fastdeploy::Runtime>(new fastdeploy::Runtime());
    if (!runtime_->Init(runtime_option)) {
        std::cerr << "--- Init FastDeploy Runitme Failed! "
                << "\n--- Model:  " << conf.model_file_path_ << std::endl;
        return -1;
    } else {
        std::cout << "--- Init FastDeploy Runitme Done! "
                << "\n--- Model:  " << conf.model_file_path_ << std::endl;
    }

    reset();
    return 0;
};

int ClsNnet::init_dict(std::string& dict_path){
    std::ifstream fp(dict_path);
    std::string line = "";
    while(getline(fp, line)){
        dict_.push_back(line);
    }
    return 0;
};

int ClsNnet::forward(const char* wav_path, int topk, char* result, int result_max_len){
#ifdef PRINT_TIME
    double duration = 0;
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
#endif
    //read wav
    WavReader wav_reader(wav_path);
    int wavform_len = wav_reader.num_samples();
    std::vector<float> wavform(wavform_len);
    memcpy(wavform.data(), wav_reader.data(), wavform_len * sizeof(float));
    waveform_float_normal(wavform);
    waveform_normal(wavform, conf_.wav_normal_, conf_.wav_normal_type_, conf_.wav_norm_mul_factor_);
#ifdef TEST_DEBUG
    {
        std::ofstream fp("cls.wavform", std::ios::out);
        for (int i = 0; i < wavform.size(); ++i) {
            fp << std::setprecision(18) << wavform[i] << " ";
        }
        fp << "\n";
    }
#endif
#ifdef PRINT_TIME
    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    printf("wav read consume: %fs\n", duration / 1000000);
#endif

#ifdef PRINT_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif

    std::vector<float> feats;
    std::unique_ptr<ppspeech::FrontendInterface> data_source(new ppspeech::DataCache());
    ppspeech::Fbank fbank(fbank_opts_, std::move(data_source));
    fbank.Accept(wavform);
    fbank.SetFinished();
    fbank.Read(&feats);

    int feat_dim = fbank_opts_.mel_opts.num_bins;
    int num_frames = feats.size() / feat_dim;
    
    for (int i = 0; i < num_frames; ++i){
        for(int j = 0; j < feat_dim; ++j){
            feats[i * feat_dim + j] = power_to_db(feats[i * feat_dim + j]);
        }            
    }
#ifdef TEST_DEBUG
    {
        std::ofstream fp("cls.feat", std::ios::out);
        for (int i = 0; i < num_frames; ++i) {
            for (int j = 0; j < feat_dim; ++j){
                fp << std::setprecision(18) << feats[i * feat_dim + j] << " ";
            }
            fp << "\n";
        }
    }
#endif
#ifdef PRINT_TIME
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    printf("extract fbank consume: %fs\n", duration / 1000000);
#endif

    // model_forward_stream(feats);

    //infer
    std::vector<float> model_out;
#ifdef PRINT_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    model_forward(feats.data(), num_frames, feat_dim, model_out);
#ifdef PRINT_TIME
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    printf("fast deploy infer consume: %fs\n", duration / 1000000);
#endif
#ifdef TEST_DEBUG
    {
        std::ofstream fp("cls.logits", std::ios::out);
        for (int i = 0; i < model_out.size(); ++i) {
            fp << std::setprecision(18) << model_out[i] << "\n";
        }
    }
#endif

    // construct result str
    ss_ << "{";
    get_topk(topk, model_out);
    ss_ << "}";

    if (result_max_len <= ss_.str().size()){
        printf("result_max_len is short than result len\n");
    }
    snprintf(result, result_max_len, "%s", ss_.str().c_str());
    return 0;
};

int ClsNnet::model_forward(const float* features, const int num_frames, const int feat_dim, std::vector<float>& model_out){
    // init input tensor shape
    fastdeploy::TensorInfo info = runtime_->GetInputInfo(0);
    info.shape = {1, num_frames, feat_dim};

    std::vector<fastdeploy::FDTensor> input_tensors(1);
    std::vector<fastdeploy::FDTensor> output_tensors(1);

    input_tensors[0].SetExternalData({1, num_frames, feat_dim}, fastdeploy::FDDataType::FP32, (void*)features);

    //get input name
    input_tensors[0].name = info.name;

    runtime_->Infer(input_tensors, &output_tensors);

    // output_tensors[0].PrintInfo();
    std::vector<int64_t> output_shape = output_tensors[0].Shape();
    model_out.resize(output_shape[0] * output_shape[1]);
    memcpy((void*)model_out.data(), output_tensors[0].Data(), output_shape[0] * output_shape[1] * sizeof(float));
    return 0;
};

int ClsNnet::model_forward_stream(std::vector<float>& feats){
    // init input tensor shape
    std::vector<fastdeploy::TensorInfo> input_infos = runtime_->GetInputInfos();
    std::vector<fastdeploy::TensorInfo> output_infos = runtime_->GetOutputInfos();

    std::vector<fastdeploy::FDTensor> input_tensors(14);
    std::vector<fastdeploy::FDTensor> output_tensors(13);
    {
    std::vector<float> feats_tmp(feats.begin(), feats.begin() + 400 * 64);
    std::vector<int> flag({0});
    std::vector<float> block1_conv1_cache(1 * 64 * 400 * 64, 0);
    std::vector<float> block1_conv2_cache(1 * 64 * 400 * 64, 0);
    std::vector<float> block2_conv1_cache(1 * 128 * 200 * 32, 0);
    std::vector<float> block2_conv2_cache(1 * 128 * 200 * 32, 0);
    std::vector<float> block3_conv1_cache(1 * 256 * 100 * 16, 0);
    std::vector<float> block3_conv2_cache(1 * 256 * 100 * 16, 0);
    std::vector<float> block4_conv1_cache(1 * 512 * 50 * 8, 0);
    std::vector<float> block4_conv2_cache(1 * 512 * 50 * 8, 0);
    std::vector<float> block5_conv1_cache(1 * 1024 * 25 * 4, 0);
    std::vector<float> block5_conv2_cache(1 * 1024 * 25 * 4, 0);
    std::vector<float> block6_conv1_cache(1 * 2048 * 12 * 2, 0);
    std::vector<float> block6_conv2_cache(1 * 2048 * 12 * 2, 0);
    input_tensors[0].name = input_infos[0].name;
    input_tensors[0].SetExternalData({1, 400, 64}, fastdeploy::FDDataType::FP32, (void*)feats_tmp.data());
    input_tensors[1].name = input_infos[1].name;
    input_tensors[1].SetExternalData({1}, fastdeploy::FDDataType::INT32, (void*)flag.data());
    input_tensors[2].name = input_infos[2].name;
    input_tensors[2].SetExternalData({1, 64, 400, 64}, fastdeploy::FDDataType::FP32, (void*)block1_conv1_cache.data());
    input_tensors[3].name = input_infos[3].name;
    input_tensors[3].SetExternalData({1, 64, 400, 64}, fastdeploy::FDDataType::FP32, (void*)block1_conv2_cache.data());
    input_tensors[4].name = input_infos[4].name;
    input_tensors[4].SetExternalData({1, 128, 200, 32}, fastdeploy::FDDataType::FP32, (void*)block2_conv1_cache.data());
    input_tensors[5].name = input_infos[5].name;
    input_tensors[5].SetExternalData({1, 128, 200, 32}, fastdeploy::FDDataType::FP32, (void*)block2_conv2_cache.data());
    input_tensors[6].name = input_infos[6].name;
    input_tensors[6].SetExternalData({1, 256, 100, 16}, fastdeploy::FDDataType::FP32, (void*)block3_conv1_cache.data());
    input_tensors[7].name = input_infos[7].name;
    input_tensors[7].SetExternalData({1, 256, 100, 16}, fastdeploy::FDDataType::FP32, (void*)block3_conv2_cache.data());
    input_tensors[8].name = input_infos[8].name;
    input_tensors[8].SetExternalData({1, 512, 50, 8}, fastdeploy::FDDataType::FP32, (void*)block4_conv1_cache.data());
    input_tensors[9].name = input_infos[9].name;
    input_tensors[9].SetExternalData({1, 512, 50, 8}, fastdeploy::FDDataType::FP32, (void*)block4_conv2_cache.data());
    input_tensors[10].name = input_infos[10].name;
    input_tensors[10].SetExternalData({1, 1024, 25, 4}, fastdeploy::FDDataType::FP32, (void*)block5_conv1_cache.data());
    input_tensors[11].name = input_infos[11].name;
    input_tensors[11].SetExternalData({1, 1024, 25, 4}, fastdeploy::FDDataType::FP32, (void*)block5_conv2_cache.data());
    input_tensors[12].name = input_infos[12].name;
    input_tensors[12].SetExternalData({1, 2048, 12, 2}, fastdeploy::FDDataType::FP32, (void*)block6_conv1_cache.data());
    input_tensors[13].name = input_infos[13].name;
    input_tensors[13].SetExternalData({1, 2048, 12, 2}, fastdeploy::FDDataType::FP32, (void*)block6_conv2_cache.data());

    std::vector<float> model_out_tmp;
#ifdef PRINT_TIME
    double duration = 0;
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
#endif
    runtime_->Infer(input_tensors, &output_tensors);
#ifdef PRINT_TIME
    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    printf("infer %d:%d consume: %fs\n", 0, 400, duration / 1000000);
#endif
    // output_tensors[0].PrintInfo();
    std::vector<int64_t> output_shape = output_tensors[0].Shape();
    model_out_tmp.resize(output_shape[0] * output_shape[1]);
    memcpy((void*)model_out_tmp.data(), output_tensors[0].Data(), output_shape[0] * output_shape[1] * sizeof(float));
#ifdef TEST_DEBUG
    {
        std::stringstream ss;
        ss << "cls.logits." << 0 << "-" << 400;
        std::ofstream fp(ss.str(), std::ios::out);
        for (int i = 0; i < model_out_tmp.size(); ++i) {
            fp << std::setprecision(18) << model_out_tmp[i] << "\n";
        }
    }
#endif
    }
    {
    std::vector<float> feats_tmp(feats.begin() + 32 * 64, feats.begin() + 432 * 64);
    std::vector<int> flag({1});
    input_tensors[0].SetExternalData({1, 400, 64}, fastdeploy::FDDataType::FP32, (void*)feats_tmp.data());
    input_tensors[1].SetExternalData({1}, fastdeploy::FDDataType::INT32, (void*)flag.data());
    input_tensors[2] = output_tensors[1];
    input_tensors[2].name = input_infos[2].name;
    input_tensors[3] = output_tensors[2];
    input_tensors[3].name = input_infos[3].name;
    input_tensors[4] = output_tensors[3];
    input_tensors[4].name = input_infos[4].name;
    input_tensors[5] = output_tensors[4];
    input_tensors[5].name = input_infos[5].name;
    input_tensors[6] = output_tensors[5];
    input_tensors[6].name = input_infos[6].name;
    input_tensors[7] = output_tensors[6];
    input_tensors[7].name = input_infos[7].name;
    input_tensors[8] = output_tensors[7];
    input_tensors[8].name = input_infos[8].name;
    input_tensors[9] = output_tensors[8];
    input_tensors[9].name = input_infos[9].name;
    input_tensors[10] = output_tensors[9];
    input_tensors[10].name = input_infos[10].name;
    input_tensors[11] = output_tensors[10];
    input_tensors[11].name = input_infos[11].name;
    input_tensors[12] = output_tensors[11];
    input_tensors[12].name = input_infos[12].name;
    input_tensors[13] = output_tensors[12];
    input_tensors[13].name = input_infos[13].name;
    std::vector<float> model_out_tmp;
#ifdef PRINT_TIME
    double duration = 0;
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
#endif
    runtime_->Infer(input_tensors, &output_tensors);
#ifdef PRINT_TIME
    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    printf("infer %d:%d consume: %fs\n", 32, 432, duration / 1000000);
#endif
    // output_tensors[0].PrintInfo();
    std::vector<int64_t> output_shape = output_tensors[0].Shape();
    model_out_tmp.resize(output_shape[0] * output_shape[1]);
    memcpy((void*)model_out_tmp.data(), output_tensors[0].Data(), output_shape[0] * output_shape[1] * sizeof(float));
#ifdef TEST_DEBUG
    {
        std::stringstream ss;
        ss << "cls.logits." << 32 << "-" << 432;
        std::ofstream fp(ss.str(), std::ios::out);
        for (int i = 0; i < model_out_tmp.size(); ++i) {
            fp << std::setprecision(18) << model_out_tmp[i] << "\n";
        }
    }
#endif
    }
    exit(1);
    return 0;
};

int ClsNnet::get_topk(int k, std::vector<float>& model_out){
    std::vector<std::pair<float, int>> sort_vec;
    for (int i = 0; i < model_out.size(); ++i){
        sort_vec.push_back({-1 * model_out[i], i});
    }
    std::sort(sort_vec.begin(), sort_vec.end());
    for (int i = 0; i < k; ++i){
        if (i != 0){
            ss_ << ",";
        }
        ss_ << "\"" << dict_[sort_vec[i].second] << "\":\"" << -1 * sort_vec[i].first << "\"";
    }
    return 0;
};

int ClsNnet::waveform_float_normal(std::vector<float>& waveform){
    int tot_samples = waveform.size();
    for (int i = 0; i < tot_samples; i++){
        waveform[i] = waveform[i] / 32768.0;
    }
    return 0;
}

int ClsNnet::waveform_normal(std::vector<float>& waveform, bool wav_normal, std::string& wav_normal_type, float wav_norm_mul_factor){
    if (wav_normal == false){
        return 0;
    }
    if (wav_normal_type == "linear"){
        float amax = INT32_MIN;
        for (int i = 0; i < waveform.size(); ++i){
            float tmp = std::abs(waveform[i]);
            amax = std::max(amax, tmp);
        }
        float factor = 1.0 / (amax + 1e-8);
        for (int i = 0; i < waveform.size(); ++i){
            waveform[i] = waveform[i] * factor * wav_norm_mul_factor;
        }
    } else if (wav_normal_type == "gaussian") {
        double sum = std::accumulate(waveform.begin(), waveform.end(), 0.0);  
        double mean =  sum / waveform.size(); //均值  

        double accum  = 0.0;  
        std::for_each (waveform.begin(), waveform.end(), [&](const double d) {  
            accum  += (d-mean)*(d-mean);  
        });  

        double stdev = sqrt(accum/(waveform.size()-1)); //方差  
        stdev = std::max(stdev, 1e-8);

        for (int i = 0; i < waveform.size(); ++i){
            waveform[i] = wav_norm_mul_factor * (waveform[i] - mean) / stdev;
        }
    } else {
        printf("don't support\n");
        return -1;
    }
    return 0;
}

float ClsNnet::power_to_db(float in, float ref_value, float amin, float top_db){
    if(amin <= 0){
        printf("amin must be strictly positive\n");
        return -1;
    };

    if(ref_value <= 0){
        printf("ref_value must be strictly positive\n");
        return -1;
    }

    float out = 10.0 * log10(std::max(amin, in));
    out -= 10.0 * log10(std::max(ref_value, amin));
    return out;
}

}