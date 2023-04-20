// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "audio_classification/nnet/panns_nnet.h"
#ifdef WITH_PROFILING
#include "kaldi/base/timer.h"
#endif

namespace ppspeech {

ClsNnet::ClsNnet() {
    // wav_reader_ = NULL;
    runtime_ = NULL;
}

void ClsNnet::Reset() {
    // wav_reader_->Clear();
    ss_.str("");
}

int ClsNnet::Init(const ClsNnetConf& conf) {
    conf_ = conf;
    // init fbank opts
    fbank_opts_.frame_opts.samp_freq = conf.samp_freq;
    fbank_opts_.frame_opts.frame_length_ms = conf.frame_length_ms;
    fbank_opts_.frame_opts.frame_shift_ms = conf.frame_shift_ms;
    fbank_opts_.mel_opts.num_bins = conf.num_bins;
    fbank_opts_.mel_opts.low_freq = conf.low_freq;
    fbank_opts_.mel_opts.high_freq = conf.high_freq;
    fbank_opts_.frame_opts.dither = conf.dither;
    fbank_opts_.use_log_fbank = false;

    // init dict
    if (conf.dict_file_path_ != "") {
        ReadFileToVector(conf.dict_file_path_, &dict_);
    }

    // init model
    fastdeploy::RuntimeOption runtime_option;

#ifdef USE_PADDLE_INFERENCE_BACKEND
    runtime_option.SetModelPath(conf.model_file_path_,
                                conf.param_file_path_,
                                fastdeploy::ModelFormat::PADDLE);
    runtime_option.UsePaddleInferBackend();
#elif defined(USE_ORT_BACKEND)
    runtime_option.SetModelPath(
        conf.model_file_path_, "", fastdeploy::ModelFormat::ONNX);  // onnx
    runtime_option.UseOrtBackend();                                 // onnx
#elif defined(USE_PADDLE_LITE_BACKEND)
    runtime_option.SetModelPath(conf.model_file_path_,
                                conf.param_file_path_,
                                fastdeploy::ModelFormat::PADDLE);
    runtime_option.UseLiteBackend();
#endif

    runtime_option.SetCpuThreadNum(conf.num_cpu_thread_);
    // runtime_option.DeletePaddleBackendPass("simplify_with_basic_ops_pass");
    runtime_ = std::unique_ptr<fastdeploy::Runtime>(new fastdeploy::Runtime());
    if (!runtime_->Init(runtime_option)) {
        std::cerr << "--- Init FastDeploy Runitme Failed! "
                  << "\n--- Model:  " << conf.model_file_path_ << std::endl;
        return -1;
    } else {
        std::cout << "--- Init FastDeploy Runitme Done! "
                  << "\n--- Model:  " << conf.model_file_path_ << std::endl;
    }

    Reset();
    return 0;
}

int ClsNnet::Forward(const char* wav_path,
                     int topk,
                     char* result,
                     int result_max_len) {
#ifdef WITH_PROFILING
    kaldi::Timer timer;
    timer.Reset();
#endif
    // read wav
    std::ifstream infile(wav_path, std::ifstream::in);
    kaldi::WaveData wave_data;
    wave_data.Read(infile);
    int32 this_channel = 0;
    kaldi::Matrix<float> wavform_kaldi = wave_data.Data();
    // only get channel 0
    int wavform_len = wavform_kaldi.NumCols();
    std::vector<float> wavform(wavform_kaldi.Data(),
                               wavform_kaldi.Data() + wavform_len);
    WaveformFloatNormal(&wavform);
    WaveformNormal(&wavform,
                   conf_.wav_normal_,
                   conf_.wav_normal_type_,
                   conf_.wav_norm_mul_factor_);
#ifdef PPS_DEBUG
    {
        std::ofstream fp("cls.wavform", std::ios::out);
        for (int i = 0; i < wavform.size(); ++i) {
            fp << std::setprecision(18) << wavform[i] << " ";
        }
        fp << "\n";
    }
#endif
#ifdef WITH_PROFILING
    printf("wav read consume: %fs\n", timer.Elapsed());
#endif

#ifdef WITH_PROFILING
    timer.Reset();
#endif

    std::vector<float> feats;
    std::unique_ptr<ppspeech::FrontendInterface> data_source(
        new ppspeech::DataCache());
    ppspeech::Fbank fbank(fbank_opts_, std::move(data_source));
    fbank.Accept(wavform);
    fbank.SetFinished();
    fbank.Read(&feats);

    int feat_dim = fbank_opts_.mel_opts.num_bins;
    int num_frames = feats.size() / feat_dim;

    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < feat_dim; ++j) {
            feats[i * feat_dim + j] = PowerTodb(feats[i * feat_dim + j]);
        }
    }
#ifdef PPS_DEBUG
    {
        std::ofstream fp("cls.feat", std::ios::out);
        for (int i = 0; i < num_frames; ++i) {
            for (int j = 0; j < feat_dim; ++j) {
                fp << std::setprecision(18) << feats[i * feat_dim + j] << " ";
            }
            fp << "\n";
        }
    }
#endif
#ifdef WITH_PROFILING
    printf("extract fbank consume: %fs\n", timer.Elapsed());
#endif

    // infer
    std::vector<float> model_out;
#ifdef WITH_PROFILING
    timer.Reset();
#endif
    ModelForward(feats.data(), num_frames, feat_dim, &model_out);
#ifdef WITH_PROFILING
    printf("fast deploy infer consume: %fs\n", timer.Elapsed());
#endif
#ifdef PPS_DEBUG
    {
        std::ofstream fp("cls.logits", std::ios::out);
        for (int i = 0; i < model_out.size(); ++i) {
            fp << std::setprecision(18) << model_out[i] << "\n";
        }
    }
#endif

    // construct result str
    ss_ << "{";
    GetTopkResult(topk, model_out);
    ss_ << "}";

    if (result_max_len <= ss_.str().size()) {
        printf("result_max_len is short than result len\n");
    }
    snprintf(result, result_max_len, "%s", ss_.str().c_str());
    return 0;
}

int ClsNnet::ModelForward(float* features,
                          const int num_frames,
                          const int feat_dim,
                          std::vector<float>* model_out) {
    // init input tensor shape
    fastdeploy::TensorInfo info = runtime_->GetInputInfo(0);
    info.shape = {1, num_frames, feat_dim};

    std::vector<fastdeploy::FDTensor> input_tensors(1);
    std::vector<fastdeploy::FDTensor> output_tensors(1);

    input_tensors[0].SetExternalData({1, num_frames, feat_dim},
                                     fastdeploy::FDDataType::FP32,
                                     static_cast<void*>(features));

    // get input name
    input_tensors[0].name = info.name;

    runtime_->Infer(input_tensors, &output_tensors);

    // output_tensors[0].PrintInfo();
    std::vector<int64_t> output_shape = output_tensors[0].Shape();
    model_out->resize(output_shape[0] * output_shape[1]);
    memcpy(static_cast<void*>(model_out->data()),
           output_tensors[0].Data(),
           output_shape[0] * output_shape[1] * sizeof(float));
    return 0;
}

int ClsNnet::GetTopkResult(int k, const std::vector<float>& model_out) {
    std::vector<float> values;
    std::vector<int> indics;
    TopK(model_out, k, &values, &indics);
    for (int i = 0; i < k; ++i) {
        if (i != 0) {
            ss_ << ",";
        }
        ss_ << "\"" << dict_[indics[i]] << "\":\"" << values[i] << "\"";
    }
    return 0;
}

}  // namespace ppspeech