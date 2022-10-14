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

namespace ppspeech {


struct ModelOptions {
    std::string model_path;
    std::string param_path;
    int thread_num;  // predictor thread pool size for ds2;
    bool use_gpu;
    bool switch_ir_optim;
    std::string input_names;
    std::string output_names;
    std::string cache_names;
    std::string cache_shape;
    bool enable_fc_padding;
    bool enable_profile;
    int subsample_rate;
    ModelOptions()
        : model_path(""),
          param_path(""),
          thread_num(1),
          use_gpu(false),
          input_names(""),
          output_names(""),
          cache_names(""),
          cache_shape(""),
          switch_ir_optim(false),
          enable_fc_padding(false),
          enable_profile(false),
          subsample_rate(0) {}

    void Register(kaldi::OptionsItf* opts) {
        opts->Register("model-path", &model_path, "model file path");
        opts->Register("model-param", &param_path, "params model file path");
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
