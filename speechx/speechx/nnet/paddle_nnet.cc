#include "nnet/paddle_nnet.h"
#include "absl/strings/str_split.h"

namespace ppspeech {

void PaddleNnet::InitCacheEncouts(const ModelOptions& opts) {
  std::vector<std::string> cache_names;
  cache_names = absl::StrSplit(opts.cache_names, ", ");
  std::vector<std::string> cache_shapes;
  cache_shapes = absl::StrSplit(opts.cache_shape, ", ");
  assert(cache_shapes.size() == cache_names.size());

  for (size_t i = 0; i < cache_shapes.size(); i++) {
    std::vector<std::string> tmp_shape;
    tmp_shape = absl::StrSplit(cache_shapes[i], "- ");
    std::vector<int> cur_shape;
    std::transform(tmp_shape.begin(), tmp_shape.end(),
                    std::back_inserter(cur_shape),
                    [](const std::string& s) {
                        return atoi(s.c_str());
                    });
    cache_names_idx_[cache_names[i]] = i;
    std::shared_ptr<Tensor<BaseFloat>> cache_eout = std::make_shared<Tensor<BaseFloat>>(cur_shape);
    cache_encouts_.push_back(cache_eout);
  }
}

PaddleNet::PaddleNnet(const ModelOptions& opts) {
    paddle_infer::Config config;
    config.SetModel(opts.model_path, opts.params_path);
    if (opts.use_gpu) {
      config.EnableUseGpu(500, 0);
    }
    config.SwitchIrOptim(opts.switch_ir_optim);
    if (opts.enbale_fc_padding) {
      config.DisableFCPadding();
    }
    if (opts.enable_profile) {
      config.EnableProfile();
    }
    pool.reset(new paddle_infer::services::PredictorPool(config, opts.thread_num));
    if (pool == nullptr) {
        LOG(ERROR) << "create the predictor pool failed";
    }
    pool_usages.resize(num_thread);
    std::fill(pool_usages.begin(), pool_usages.end(), false);
    LOG(INFO) << "load paddle model success";

    LOG(INFO) << "start to check the predictor input and output names";
    LOG(INFO) << "input names: " << opts.input_names;
    LOG(INFO) << "output names: " << opts.output_names;
    vector<string> input_names_vec = absl::StrSplit(opts.input_names, ", ");
    vector<string> output_names_vec = absl::StrSplit(opts.output_names, ", ");
    paddle_infer::Predictor* predictor = get_predictor();
        
    std::vector<std::string> model_input_names = predictor->GetInputNames();
    assert(input_names_vec.size() == model_input_names.size());
    for (size_t i = 0; i < model_input_names.size(); i++) {
        assert(input_names_vec[i] == model_input_names[i]);
    }

    std::vector<std::string> model_output_names = predictor->GetOutputNames();
    assert(output_names_vec.size() == model_output_names.size());
    for (size_t i = 0;i < output_names_vec.size(); i++) {
        assert(output_names_vec[i] == model_output_names[i]);
    }
    release_predictor(predictor);

    InitCacheEncouts(opts);
}

paddle_infer::Predictor* PaddleNnet::get_predictor() {
    LOG(INFO) << "attempt to get a new predictor instance " << std::endl;
    paddle_infer::Predictor* predictor = nullptr;
    std::lock_guard<std::mutex> guard(pool_mutex);
    int pred_id = 0;

    while (pred_id < pool_usages.size()) {
        if (pool_usages[pred_id] == false) {
            predictor = pool->Retrive(pred_id);
            break;
        }
        ++pred_id;
    }

    if (predictor) {
        pool_usages[pred_id] = true;
        predictor_to_thread_id[predictor] = pred_id;
        LOG(INFO) << pred_id << " predictor create success";
    } else {
        LOG(INFO) << "Failed to get predictor from pool !!!";
    }

    return predictor;
}

int PaddleNnet::ReleasePredictor(paddle_infer::Predictor* predictor) {
    LOG(INFO) << "attempt to releae a predictor";
    std::lock_guard<std::mutex> guard(pool_mutex);
    auto iter = predictor_to_thread_id.find(predictor);

    if (iter == predictor_to_thread_id.end()) {
        LOG(INFO) << "there is no such predictor";
        return 0;
    }

    LOG(INFO) << iter->second << " predictor will be release";
    pool_usages[iter->second] = false;
    predictor_to_thread_id.erase(predictor);
    LOG(INFO) << "release success";
    return 0;
}



shared_ptr<Tensor<BaseFloat>> PaddleNnet::GetCacheEncoder(const string& name) {
  auto iter = cache_names_idx_.find(name);
  if (iter == cache_names_idx_.end()) {
    return nullptr;
  }
  assert(iter->second < cache_encouts_.size());
  return cache_encouts_[iter->second].get(); 
}

void PaddleNet::FeedForward(const Matrix<BaseFloat>& features, Matrix<BaseFloat>* inferences) const {
    
    // 1. 得到所有的 input tensor 的名称
    int row = features.NumRows();
    int col = features.NumCols();
    std::vector<std::string> input_names = predictor->GetInputNames();
    std::vector<std::string> output_names = predictor->GetOutputNames();
    LOG(INFO) << "feat info: row=" << row << ", col=" << col;

    std::unique_ptr<paddle_infer::Tensor> input_tensor = predictor->GetInputHandle(input_names[0]);
    std::vector<int> INPUT_SHAPE = {1, row, col};
    input_tensor->Reshape(INPUT_SHAPE);
    input_tensor->CopyFromCpu(features.Data());
    // 3. 输入每个音频帧数
    std::unique_ptr<paddle_infer::Tensor> input_len = predictor->GetInputHandle(input_names[1]);
    std::vector<int> input_len_size = {1};
    input_len->Reshape(input_len_size);
    std::vector<int64_t> audio_len;
    audio_len.push_back(row);
    input_len->CopyFromCpu(audio_len.data());
    // 输入流式的缓存数据
    std::unique_ptr<paddle_infer::Tensor> h_box = predictor->GetInputHandle(input_names[2]);
    share_ptr<Tensor<BaseFloat>> h_cache = GetCacheEncoder(input_names[2]));
    h_box->Reshape(h_cache->get_shape());
    h_box->CopyFromCpu(h_cache->get_data().data());
    std::unique_ptr<paddle_infer::Tensor> c_box = predictor->GetInputHandle(input_names[3]);
    share_ptr<Tensor<float>> c_cache = GetCacheEncoder(input_names[3]);
    c_box->Reshape(c_cache->get_shape());
    c_box->CopyFromCpu(c_cache->get_data().data());
    std::thread::id this_id = std::this_thread::get_id();
    LOG(INFO) << this_id << " start to compute the probability";
    bool success = predictor->Run();

    if (success == false) {
        LOG(INFO) << "predictor run occurs error";
    }

    LOG(INFO) << "get the model success";
    std::unique_ptr<paddle_infer::Tensor> h_out = predictor->GetOutputHandle(output_names[2]);
    assert(h_cache->get_shape() == h_out->shape());
    h_out->CopyToCpu(h_cache->get_data().data());
    std::unique_ptr<paddle_infer::Tensor> c_out = predictor->GetOutputHandle(output_names[3]);
    assert(c_cache->get_shape() == c_out->shape());
    c_out->CopyToCpu(c_cache->get_data().data());
    // 5. 得到最后的输出结果
    std::unique_ptr<paddle_infer::Tensor> output_tensor =
        predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_tensor->shape();
    row = output_shape[1];
    col = output_shape[2];
    inference.Resize(row, col);
    output_tensor->CopyToCpu(inference.Data());
}

} // namespace ppspeech           