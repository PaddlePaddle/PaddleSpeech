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
#include "frontend/audio/frontend_itf.h"

namespace ppspeech {

struct AssemblerOptions {
    // refer:https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/s2t/exps/deepspeech2/model.py
    // the nnet batch forward
    int32 receptive_filed_length;
    int32 subsampling_rate;
    int32 nnet_decoder_chunk;
    
    AssemblerOptions()
        : receptive_filed_length(1),
          subsampling_rate(1),
          nnet_decoder_chunk(1) {}
};

class Assembler : public FrontendInterface {
  public:
    explicit Assembler(
        AssemblerOptions opts,
        std::unique_ptr<FrontendInterface> base_extractor = NULL);

    // Feed feats or waves
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs);

    // feats size = num_frames * feat_dim
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feats);

    // feat dim
    virtual size_t Dim() const { return dim_; }

    virtual void SetFinished() {
        base_extractor_->SetFinished();
    }

    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }

    virtual void Reset() {
        base_extractor_->Reset();
    }

  private:
    bool Compute(kaldi::Vector<kaldi::BaseFloat>* feats);

    int32 dim_;
    int32 frame_chunk_size_;    // window
    int32 frame_chunk_stride_;  // stride
    int32 receptive_filed_length_;
    std::queue<kaldi::Vector<kaldi::BaseFloat>> feature_cache_;
    std::unique_ptr<FrontendInterface> base_extractor_;
    DISALLOW_COPY_AND_ASSIGN(Assembler);
};

}  // namespace ppspeech
