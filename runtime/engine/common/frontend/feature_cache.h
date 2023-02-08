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

class FeatureCache : public FrontendInterface {
  public:
    explicit FeatureCache(
        size_t max_size = kint16max,
        std::unique_ptr<FrontendInterface> base_extractor = NULL);

    // Feed feats or waves
    virtual void Accept(const std::vector<kaldi::BaseFloat>& inputs);

    // feats size = num_frames * feat_dim
    virtual bool Read(std::vector<kaldi::BaseFloat>* feats);

    // feat dim
    virtual size_t Dim() const { return dim_; }

    virtual void SetFinished() {
        std::unique_lock<std::mutex> lock(mutex_);
        LOG(INFO) << "set finished";
        // read the last chunk data
        Compute();
        base_extractor_->SetFinished();
        LOG(INFO) << "compute last feats done.";
    }

    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }

    void Reset() override {
        std::queue<std::vector<BaseFloat>> empty;
        std::swap(cache_, empty);
        nframe_ = 0;
        base_extractor_->Reset();
        VLOG(3) << "feature cache reset: cache size: " << cache_.size();
    }

  private:
    bool Compute();

    int32 dim_;
    size_t max_size_;  // cache capacity
    std::unique_ptr<FrontendInterface> base_extractor_;

    std::queue<std::vector<BaseFloat>> cache_;  // feature cache
    std::mutex mutex_;

    int32 nframe_;  // num of feature computed
    DISALLOW_COPY_AND_ASSIGN(FeatureCache);
};

}  // namespace ppspeech
