// Copyright 2022 Horizon Robotics. All Rights Reserved.
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

// modified from
// https://github.com/wenet-e2e/wenet/blob/main/runtime/core/decoder/asr_model.h
#pragma once

#include "base/common.h"
#include "matrix/kaldi-matrix.h"
#include "nnet/nnet_itf.h"
#include "paddle/extension.h"
#include "paddle/jit/all.h"
#include "paddle/phi/api/all.h"

namespace ppspeech {


class U2NnetBase : public NnetBase {
  public:
    virtual int Context() const { return right_context_ + 1; }
    virtual int RightContext() const { return right_context_; }

    virtual int EOS() const { return eos_; }
    virtual int SOS() const { return sos_; }
    virtual int IsBidecoder() const { return is_bidecoder_; }
    // current offset in decoder frame
    virtual int Offset() const { return offset_; }
    virtual void SetChunkSize(int chunk_size) { chunk_size_ = chunk_size; }
    virtual void SetNumLeftChunks(int num_left_chunks) {
        num_left_chunks_ = num_left_chunks;
    }

    virtual std::shared_ptr<NnetBase> Clone() const = 0;

  protected:
    virtual void ForwardEncoderChunkImpl(
        const std::vector<kaldi::BaseFloat>& chunk_feats,
        const int32& feat_dim,
        std::vector<kaldi::BaseFloat>* ctc_probs,
        int32* vocab_dim) = 0;

  protected:
    // model specification
    int right_context_{0};

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
    explicit U2Nnet(const ModelOptions& opts);
    U2Nnet(const U2Nnet& other);

    void FeedForward(const std::vector<kaldi::BaseFloat>& features,
                     const int32& feature_dim,
                     NnetOut* out) override;

    void Reset() override;

    bool IsLogProb() override { return true; }

    void Dim();

    void LoadModel(const std::string& model_path_w_prefix);
    void Warmup();

    std::shared_ptr<paddle::jit::Layer> model() const { return model_; }

    std::shared_ptr<NnetBase> Clone() const override;

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
    void FeedEncoderOuts(const paddle::Tensor& encoder_out);

    void EncoderOuts(
        std::vector<std::vector<kaldi::BaseFloat>>* encoder_out) const;

    ModelOptions opts_; // hack, fix later
  private:

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
    float cost_time_ = 0.0;
};

}  // namespace ppspeech