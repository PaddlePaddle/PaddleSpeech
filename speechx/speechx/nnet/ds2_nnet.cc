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

#include "nnet/ds2_nnet.h"
#include "absl/strings/str_split.h"

namespace ppspeech {

using std::vector;
using std::string;
using std::shared_ptr;
using kaldi::Matrix;
using kaldi::Vector;

void PaddleNnet::InitCacheEncouts(const ModelOptions& opts) {
    std::vector<std::string> cache_names;
    cache_names = absl::StrSplit(opts.cache_names, ",");
    std::vector<std::string> cache_shapes;
    cache_shapes = absl::StrSplit(opts.cache_shape, ",");
    assert(cache_shapes.size() == cache_names.size());

    cache_encouts_.clear();
    cache_names_idx_.clear();
    for (size_t i = 0; i < cache_shapes.size(); i++) {
        std::vector<std::string> tmp_shape;
        tmp_shape = absl::StrSplit(cache_shapes[i], "-");
        std::vector<int> cur_shape;
        std::transform(tmp_shape.begin(),
                       tmp_shape.end(),
                       std::back_inserter(cur_shape),
                       [](const std::string& s) { return atoi(s.c_str()); });
        cache_names_idx_[cache_names[i]] = i;
        std::shared_ptr<Tensor<BaseFloat>> cache_eout =
            std::make_shared<Tensor<BaseFloat>>(cur_shape);
        cache_encouts_.push_back(cache_eout);
    }
}

PaddleNnet::PaddleNnet(const ModelOptions& opts) : opts_(opts) {
    subsampling_rate_ = opts.subsample_rate;
    paddle_infer::Config config;
    config.SetModel(opts.model_path, opts.param_path);
    if (opts.use_gpu) {
        config.EnableUseGpu(500, 0);
    }
    config.SwitchIrOptim(opts.switch_ir_optim);
    if (opts.enable_fc_padding == false) {
        config.DisableFCPadding();
    }
    if (opts.enable_profile) {
        config.EnableProfile();
    }
    pool.reset(
        new paddle_infer::services::PredictorPool(config, opts.thread_num));
    if (pool == nullptr) {
        LOG(ERROR) << "create the predictor pool failed";
    }
    pool_usages.resize(opts.thread_num);
    std::fill(pool_usages.begin(), pool_usages.end(), false);
    LOG(INFO) << "load paddle model success";

    LOG(INFO) << "start to check the predictor input and output names";
    LOG(INFO) << "input names: " << opts.input_names;
    LOG(INFO) << "output names: " << opts.output_names;
    vector<string> input_names_vec = absl::StrSplit(opts.input_names, ",");
    vector<string> output_names_vec = absl::StrSplit(opts.output_names, ",");

    paddle_infer::Predictor* predictor = GetPredictor();

    std::vector<std::string> model_input_names = predictor->GetInputNames();
    assert(input_names_vec.size() == model_input_names.size());
    for (size_t i = 0; i < model_input_names.size(); i++) {
        assert(input_names_vec[i] == model_input_names[i]);
    }

    std::vector<std::string> model_output_names = predictor->GetOutputNames();
    assert(output_names_vec.size() == model_output_names.size());
    for (size_t i = 0; i < output_names_vec.size(); i++) {
        assert(output_names_vec[i] == model_output_names[i]);
    }

    ReleasePredictor(predictor);
    InitCacheEncouts(opts);
}

void PaddleNnet::Reset() { InitCacheEncouts(opts_); }

paddle_infer::Predictor* PaddleNnet::GetPredictor() {
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
    } else {
        LOG(INFO) << "Failed to get predictor from pool !!!";
    }

    return predictor;
}

int PaddleNnet::ReleasePredictor(paddle_infer::Predictor* predictor) {
    std::lock_guard<std::mutex> guard(pool_mutex);
    auto iter = predictor_to_thread_id.find(predictor);

    if (iter == predictor_to_thread_id.end()) {
        LOG(INFO) << "there is no such predictor";
        return 0;
    }

    pool_usages[iter->second] = false;
    predictor_to_thread_id.erase(predictor);
    return 0;
}

shared_ptr<Tensor<BaseFloat>> PaddleNnet::GetCacheEncoder(const string& name) {
    auto iter = cache_names_idx_.find(name);
    if (iter == cache_names_idx_.end()) {
        return nullptr;
    }
    assert(iter->second < cache_encouts_.size());
    return cache_encouts_[iter->second];
}

void PaddleNnet::FeedForward(const Vector<BaseFloat>& features,
                             const int32& feature_dim,
                             NnetOut* out) {
    paddle_infer::Predictor* predictor = GetPredictor();

    int feat_row = features.Dim() / feature_dim;

    std::vector<std::string> input_names = predictor->GetInputNames();
    std::vector<std::string> output_names = predictor->GetOutputNames();

    // feed inputs
    std::unique_ptr<paddle_infer::Tensor> input_tensor =
        predictor->GetInputHandle(input_names[0]);
    std::vector<int> INPUT_SHAPE = {1, feat_row, feature_dim};
    input_tensor->Reshape(INPUT_SHAPE);
    input_tensor->CopyFromCpu(features.Data());

    std::unique_ptr<paddle_infer::Tensor> input_len =
        predictor->GetInputHandle(input_names[1]);
    std::vector<int> input_len_size = {1};
    input_len->Reshape(input_len_size);
    std::vector<int64_t> audio_len;
    audio_len.push_back(feat_row);
    input_len->CopyFromCpu(audio_len.data());

    std::unique_ptr<paddle_infer::Tensor> state_h =
        predictor->GetInputHandle(input_names[2]);
    shared_ptr<Tensor<BaseFloat>> h_cache = GetCacheEncoder(input_names[2]);
    state_h->Reshape(h_cache->get_shape());
    state_h->CopyFromCpu(h_cache->get_data().data());

    std::unique_ptr<paddle_infer::Tensor> state_c =
        predictor->GetInputHandle(input_names[3]);
    shared_ptr<Tensor<float>> c_cache = GetCacheEncoder(input_names[3]);
    state_c->Reshape(c_cache->get_shape());
    state_c->CopyFromCpu(c_cache->get_data().data());

    // forward
    bool success = predictor->Run();

    if (success == false) {
        LOG(INFO) << "predictor run occurs error";
    }

    // fetch outpus
    std::unique_ptr<paddle_infer::Tensor> h_out =
        predictor->GetOutputHandle(output_names[2]);
    assert(h_cache->get_shape() == h_out->shape());
    h_out->CopyToCpu(h_cache->get_data().data());

    std::unique_ptr<paddle_infer::Tensor> c_out =
        predictor->GetOutputHandle(output_names[3]);
    assert(c_cache->get_shape() == c_out->shape());
    c_out->CopyToCpu(c_cache->get_data().data());

    std::unique_ptr<paddle_infer::Tensor> output_tensor =
        predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_tensor->shape();
    int32 row = output_shape[1];
    int32 col = output_shape[2];


    // inferences->Resize(row * col);
    // *inference_dim = col;
    out->logprobs.Resize(row*col);
    out->vocab_dim = col;
    output_tensor->CopyToCpu(out->logprobs.Data());

    ReleasePredictor(predictor);
}

}  // namespace ppspeech