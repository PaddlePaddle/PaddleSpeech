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
#include <numeric>

#include "base/common.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "nnet/nnet_itf.h"
#include "paddle_inference_api.h"

namespace ppspeech {


template <typename T>
class Tensor {
  public:
    Tensor() {}
    Tensor(const std::vector<int>& shape) : _shape(shape) {
        int neml = std::accumulate(
            _shape.begin(), _shape.end(), 1, std::multiplies<int>());
        LOG(INFO) << "Tensor neml: " << neml;
        _data.resize(neml, 0);
    }

    void reshape(const std::vector<int>& shape) {
        _shape = shape;
        int neml = std::accumulate(
            _shape.begin(), _shape.end(), 1, std::multiplies<int>());
        _data.resize(neml, 0);
    }

    const std::vector<int>& get_shape() const { return _shape; }
    std::vector<T>& get_data() { return _data; }

  private:
    std::vector<int> _shape;
    std::vector<T> _data;
};

class PaddleNnet : public NnetBase {
  public:
    PaddleNnet(const ModelOptions& opts);

    void FeedForward(const kaldi::Vector<kaldi::BaseFloat>& features,
                     const int32& feature_dim,
                     NnetOut* out) override;

    void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                            float reverse_weight,
                            std::vector<float>* rescoring_score) override {
        VLOG(2) << "deepspeech2 not has AttentionRescoring.";
    }

    void Dim();

    void Reset() override;

    bool IsLogProb() override { return false; }


    std::shared_ptr<Tensor<kaldi::BaseFloat>> GetCacheEncoder(
        const std::string& name);

    void InitCacheEncouts(const ModelOptions& opts);

    void EncoderOuts(std::vector<kaldi::Vector<kaldi::BaseFloat>>* encoder_out)
        const override {}

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
