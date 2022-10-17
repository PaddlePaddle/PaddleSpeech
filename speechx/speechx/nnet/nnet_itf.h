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

#include "base/basic_types.h"
#include "kaldi/base/kaldi-types.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "kaldi/util/options-itf.h"

DECLARE_int32(subsampling_rate);
DECLARE_string(model_path);
DECLARE_string(param_path);
DECLARE_string(model_input_names);
DECLARE_string(model_output_names);
DECLARE_string(model_cache_names);
DECLARE_string(model_cache_shapes);

namespace ppspeech {

struct ModelOptions {
    // common
    int subsample_rate{1};
    int thread_num{1};  // predictor thread pool size for ds2;
    bool use_gpu{false};
    std::string model_path;

    std::string param_path;

    // ds2 for inference
    std::string input_names{};
    std::string output_names{};
    std::string cache_names{};
    std::string cache_shape{};
    bool switch_ir_optim{false};
    bool enable_fc_padding{false};
    bool enable_profile{false};

    static ModelOptions InitFromFlags() {
        ModelOptions opts;
        opts.subsample_rate = FLAGS_subsampling_rate;
        LOG(INFO) << "subsampling rate: " << opts.subsample_rate;
        opts.model_path = FLAGS_model_path;
        LOG(INFO) << "model path: " << opts.model_path;

        opts.param_path = FLAGS_param_path;
        LOG(INFO) << "param path: " << opts.param_path;

        LOG(INFO) << "DS2 param: ";
        opts.cache_names = FLAGS_model_cache_names;
        LOG(INFO) << "  cache names: " << opts.cache_names;
        opts.cache_shape = FLAGS_model_cache_shapes;
        LOG(INFO) << "  cache shape: " << opts.cache_shape;
        opts.input_names = FLAGS_model_input_names;
        LOG(INFO) << "  input names: " << opts.input_names;
        opts.output_names = FLAGS_model_output_names;
        LOG(INFO) << "  output names: " << opts.output_names;
        return opts;
    }
};

struct NnetOut {
    // nnet out. maybe logprob or prob. Almost time this is logprob.
    kaldi::Vector<kaldi::BaseFloat> logprobs;
    int32 vocab_dim;

    // nnet state. Only using in Attention model.
    std::vector<std::vector<kaldi::BaseFloat>> encoder_outs;

    NnetOut() : logprobs({}), vocab_dim(-1), encoder_outs({}) {}
};


class NnetInterface {
  public:
    virtual ~NnetInterface() {}

    // forward feat with nnet.
    // nnet do not cache feats, feats cached by frontend.
    // nnet cache model state, i.e. encoder_outs, att_cache, cnn_cache,
    // frame_offset.
    virtual void FeedForward(const kaldi::Vector<kaldi::BaseFloat>& features,
                             const int32& feature_dim,
                             NnetOut* out) = 0;

    virtual void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                    float reverse_weight,
                                    std::vector<float>* rescoring_score) = 0;

    // reset nnet state, e.g. nnet_logprob_cache_, offset_, encoder_outs_.
    virtual void Reset() = 0;

    // true, nnet output is logprob; otherwise is prob,
    virtual bool IsLogProb() = 0;

    int SubsamplingRate() const { return subsampling_rate_; }

    // using to get encoder outs. e.g. seq2seq with Attention model.
    virtual void EncoderOuts(
        std::vector<kaldi::Vector<kaldi::BaseFloat>>* encoder_out) const = 0;

  protected:
    int subsampling_rate_{1};
};

}  // namespace ppspeech
