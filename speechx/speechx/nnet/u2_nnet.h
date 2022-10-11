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

#include "base/common.h"
#include "kaldi/matrix/kaldi-matrix.h"

#include "kaldi/util/options-itf.h"
#include "nnet/nnet_itf.h"

#include "paddle/extension.h"
#include "paddle/jit/all.h"
#include "paddle/phi/api/all.h"

namespace ppspeech {

struct U2ModelOptions {
    std::string model_path;
    int thread_num;
    bool use_gpu;
    U2ModelOptions() : model_path(""), thread_num(1), use_gpu(false) {}

    void Register(kaldi::OptionsItf* opts) {
        opts->Register("model-path", &model_path, "model file path");
        opts->Register("thread-num", &thread_num, "thread num");
        opts->Register("use-gpu", &use_gpu, "if use gpu");
    }
};


class U2NnetBase : public NnetInterface {
  public:
    virtual int context() const { return right_context_ + 1; }
    virtual int right_context() const { return right_context_; }
    virtual int subsampling_rate() const { return subsampling_rate_; }
    virtual int eos() const { return eos_; }
    virtual int sos() const { return sos_; }
    virtual int is_bidecoder() const { return is_bidecoder_; }
    // current offset in decoder frame
    virtual int offset() const { return offset_; }
    virtual void set_chunk_size(int chunk_size) { chunk_size_ = chunk_size; }
    virtual void set_num_left_chunks(int num_left_chunks) {
        num_left_chunks_ = num_left_chunks;
    }
    // start: false, it is the start chunk of one sentence, else true
    virtual int num_frames_for_chunk(bool start) const;

    virtual std::shared_ptr<NnetInterface> Copy() const = 0;

    virtual void ForwardEncoderChunk(
        const std::vector<kaldi::BaseFloat>& chunk_feats,
        const int32& feat_dim,
        std::vector<kaldi::BaseFloat>* ctc_probs,
        int32* vocab_dim);

    virtual void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                    float reverse_weight,
                                    std::vector<float>* rescoring_score) = 0;

  protected:
    virtual void ForwardEncoderChunkImpl(
        const std::vector<kaldi::BaseFloat>& chunk_feats,
        const int32& feat_dim,
        std::vector<kaldi::BaseFloat>* ctc_probs,
        int32* vocab_dim) = 0;

    virtual void CacheFeature(const std::vector<kaldi::BaseFloat>& chunk_feats,
                              int32 feat_dim);

  protected:
    // model specification
    int right_context_{0};
    int subsampling_rate_{1};

    int sos_{0};
    int eos_{0};

    bool is_bidecoder_{false};

    int chunk_size_{16};  // num of decoder frames. If chunk_size > 0, streaming
                          // case. Otherwise, none streaming case
    int num_left_chunks_{-1};  // -1 means all left chunks

    // asr decoder state, not used in nnet
    int offset_{0};  // current offset in encoder output time stamp. Used by
                     // position embedding.
    std::vector<std::vector<float>> cached_feats_{};  // features cache
};


class U2Nnet : public U2NnetBase {
  public:
    U2Nnet(const U2ModelOptions& opts);
    U2Nnet(const U2Nnet& other);

    void FeedForward(const kaldi::Vector<kaldi::BaseFloat>& features,
                     const int32& feature_dim,
                     NnetOut* out) override;

    void Reset() override;

    void Dim();

    void LoadModel(const std::string& model_path_w_prefix);
    void Warmup();

    std::shared_ptr<paddle::jit::Layer> model() const { return model_; }

    std::shared_ptr<NnetInterface> Copy() const override;

    void ForwardEncoderChunkImpl(
        const std::vector<kaldi::BaseFloat>& chunk_feats,
        const int32& feat_dim,
        std::vector<kaldi::BaseFloat>* ctc_probs,
        int32* vocab_dim) override;

    float ComputePathScore(const paddle::Tensor& prob,
                           const std::vector<int>& hyp,
                           int eos);

    void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                            float reverse_weight,
                            std::vector<float>* rescoring_score) override;

    // debug
    void FeedEncoderOuts(paddle::Tensor& encoder_out);

    const std::vector<paddle::Tensor>& EncoderOuts() const {return encoder_outs_; }

  private:
    U2ModelOptions opts_;

    phi::Place dev_;
    std::shared_ptr<paddle::jit::Layer> model_{nullptr};
    std::vector<paddle::Tensor> encoder_outs_;
    // transformer/conformer attention cache
    paddle::Tensor att_cache_ = paddle::full({0, 0, 0, 0}, 0.0);
    // conformer-only conv_module cache
    paddle::Tensor cnn_cache_ = paddle::full({0, 0, 0, 0}, 0.0);

    paddle::jit::Function forward_encoder_chunk_;
    paddle::jit::Function forward_attention_decoder_;
    paddle::jit::Function ctc_activation_;
};

}  // namespace ppspeech