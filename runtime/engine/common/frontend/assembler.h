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
#include "frontend/frontend_itf.h"

namespace ppspeech {

struct AssemblerOptions {
    // refer:https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/s2t/exps/deepspeech2/model.py
    // the nnet batch forward
    int32 receptive_filed_length{1};
    int32 subsampling_rate{1};
    int32 nnet_decoder_chunk{1};
    bool fill_zero{false};  // whether fill zero when last chunk is not equal to
                            // frame_chunk_size_
};

class Assembler : public FrontendInterface {
  public:
    explicit Assembler(
        AssemblerOptions opts,
        std::unique_ptr<FrontendInterface> base_extractor = NULL);

    // Feed feats or waves
    void Accept(const std::vector<kaldi::BaseFloat>& inputs) override;

    // feats size = num_frames * feat_dim
    bool Read(std::vector<kaldi::BaseFloat>* feats) override;

    // feat dim
    size_t Dim() const override { return dim_; }

    void SetFinished() override { base_extractor_->SetFinished(); }

    bool IsFinished() const override { return base_extractor_->IsFinished(); }

    void Reset() override;

  private:
    bool Compute(std::vector<kaldi::BaseFloat>* feats);

    bool fill_zero_{false};

    int32 dim_;                 // feat dim
    int32 frame_chunk_size_;    // window
    int32 frame_chunk_stride_;  // stride
    int32 cache_size_;          // window - stride
    int32 receptive_filed_length_;
    std::queue<std::vector<kaldi::BaseFloat>> feature_cache_;
    std::unique_ptr<FrontendInterface> base_extractor_;

    int32 nframes_;  // num frame computed
    DISALLOW_COPY_AND_ASSIGN(Assembler);
};

}  // namespace ppspeech
