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

struct DispenserOptions {
    int32 frame_chunk_size;
    int32 frame_chunk_stride;
    
    DispenserOptions()
        : frame_chunk_size(1),
          frame_chunk_stride(1) {}
};

class Dispenser : public FrontendInterface {
  public:
    explicit Dispenser(
        DispenserOptions opts,
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
    std::queue<kaldi::Vector<kaldi::BaseFloat>> feature_cache_;
    std::unique_ptr<FrontendInterface> base_extractor_;
    DISALLOW_COPY_AND_ASSIGN(Dispenser);
};

}  // namespace ppspeech
