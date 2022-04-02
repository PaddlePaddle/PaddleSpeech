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


#pragma once


#include "kaldi/matrix/kaldi-matrix.h"
#include "kaldi/util/options-itf.h"

#include "base/common.h"
#include "nnet/nnet_itf.h"
#include "paddle_inference_api.h"

#include <numeric>

namespace ppspeech {

struct ModelOptions {
    std::string model_path;
    std::string params_path;
    int thread_num;
    bool use_gpu;
    bool switch_ir_optim;
    std::string input_names;
    std::string output_names;
    std::string cache_names;
    std::string cache_shape;
    bool enable_fc_padding;
    bool enable_profile;
    ModelOptions()
        : model_path("avg_1.jit.pdmodel"),
          params_path("avg_1.jit.pdiparams"),
          thread_num(2),
          use_gpu(false),
          input_names(
              "audio_chunk,audio_chunk_lens,chunk_state_h_box,chunk_state_c_"
              "box"),
          output_names(
              "save_infer_model/scale_0.tmp_1,save_infer_model/"
              "scale_1.tmp_1,save_infer_model/scale_2.tmp_1,save_infer_model/"
              "scale_3.tmp_1"),
          cache_names("chunk_state_h_box,chunk_state_c_box"),
          cache_shape("3-1-1024,3-1-1024"),
          switch_ir_optim(false),
          enable_fc_padding(false),
          enable_profile(false) {}

    void Register(kaldi::OptionsItf* opts) {
        opts->Register("model-path", &model_path, "model file path");
        opts->Register("model-params", &params_path, "params model file path");
        opts->Register("thread-num", &thread_num, "thread num");
        opts->Register("use-gpu", &use_gpu, "if use gpu");
        opts->Register("input-names", &input_names, "paddle input names");
        opts->Register("output-names", &output_names, "paddle output names");
        opts->Register("cache-names", &cache_names, "cache names");
        opts->Register("cache-shape", &cache_shape, "cache shape");
        opts->Register("switch-ir-optiom",
                       &switch_ir_optim,
                       "paddle SwitchIrOptim option");
        opts->Register("enable-fc-padding",
                       &enable_fc_padding,
                       "paddle EnableFCPadding option");
        opts->Register(
            "enable-profile", &enable_profile, "paddle EnableProfile option");
    }
};

template <typename T>
class Tensor {
  public:
    Tensor() {}
    Tensor(const std::vector<int>& shape) : _shape(shape) {
        int data_size = std::accumulate(
            _shape.begin(), _shape.end(), 1, std::multiplies<int>());
        LOG(INFO) << "data size: " << data_size;
        _data.resize(data_size, 0);
    }
    void reshape(const std::vector<int>& shape) {
        _shape = shape;
        int data_size = std::accumulate(
            _shape.begin(), _shape.end(), 1, std::multiplies<int>());
        _data.resize(data_size, 0);
    }
    const std::vector<int>& get_shape() const { return _shape; }
    std::vector<T>& get_data() { return _data; }

  private:
    std::vector<int> _shape;
    std::vector<T> _data;
};

class PaddleNnet : public NnetInterface {
  public:
    PaddleNnet(const ModelOptions& opts);
    virtual void FeedForward(const kaldi::Vector<kaldi::BaseFloat>& features,
                             int32 feature_dim,
                             kaldi::Vector<kaldi::BaseFloat>* inferences,
                             int32* inference_dim);
    void Dim();
    virtual void Reset();
    std::shared_ptr<Tensor<kaldi::BaseFloat>> GetCacheEncoder(
        const std::string& name);
    void InitCacheEncouts(const ModelOptions& opts);

  private:
    paddle_infer::Predictor* GetPredictor();
    int ReleasePredictor(paddle_infer::Predictor* predictor);

    std::unique_ptr<paddle_infer::services::PredictorPool> pool;
    std::vector<bool> pool_usages;
    std::mutex pool_mutex;
    std::map<paddle_infer::Predictor*, int> predictor_to_thread_id;
    std::map<std::string, int> cache_names_idx_;
    std::vector<std::shared_ptr<Tensor<kaldi::BaseFloat>>> cache_encouts_;
    ModelOptions opts_;

  public:
    DISALLOW_COPY_AND_ASSIGN(PaddleNnet);
};

}  // namespace ppspeech
