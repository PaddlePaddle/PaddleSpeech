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

struct FeatureCacheOptions {
    int32 max_size;
    int32 frame_chunk_size;
    int32 frame_chunk_stride;
    FeatureCacheOptions()
        : max_size(kint16max), frame_chunk_size(1), frame_chunk_stride(1) {}
};

class FeatureCache : public FrontendInterface {
  public:
    explicit FeatureCache(
        FeatureCacheOptions opts,
        std::unique_ptr<FrontendInterface> base_extractor = NULL);

    // Feed feats or waves
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs);

    // feats size = num_frames * feat_dim
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feats);

    // feat dim
    virtual size_t Dim() const { return dim_; }

    virtual void SetFinished() {
        base_extractor_->SetFinished();
        // read the last chunk data
        Compute();
    }

    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }

    virtual void Reset() {
        base_extractor_->Reset();
        while (!cache_.empty()) {
            cache_.pop();
        }
    }

  private:
    bool Compute();

    int32 dim_;
    size_t max_size_;
    int32 frame_chunk_size_;
    int32 frame_chunk_stride_;

    kaldi::Vector<kaldi::BaseFloat> remained_feature_;
    std::unique_ptr<FrontendInterface> base_extractor_;
    std::mutex mutex_;
    std::queue<kaldi::Vector<BaseFloat>> cache_;
    std::condition_variable ready_feed_condition_;
    std::condition_variable ready_read_condition_;

    // DISALLOW_COPY_AND_ASSGIN(FeatureCache);
};

}  // namespace ppspeech
