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
#include "frontend/feature_extractor_interface.h"

namespace ppspeech {

class FeatureCache : public FeatureExtractorInterface {
  public:
    explicit FeatureCache(
        int32 max_size = kint16max,
        std::unique_ptr<FeatureExtractorInterface> base_extractor = NULL);
    virtual void Accept(
        const kaldi::VectorBase<kaldi::BaseFloat>& inputs);
    // output_feats dim = num_frames * feature_dim
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* output_feats);
    // feature cache only cache feature which from base extractor
    virtual size_t Dim() const { return base_extractor_->Dim(); }
    virtual void SetFinished() {
        base_extractor_->SetFinished();
        // read the last chunk data
        Compute();
    }
    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }

  private:
    bool Compute();

    bool finished_;
    std::mutex mutex_;
    size_t max_size_;
    std::queue<kaldi::Vector<BaseFloat>> cache_;
    std::unique_ptr<FeatureExtractorInterface> base_extractor_;
    std::condition_variable ready_feed_condition_;
    std::condition_variable ready_read_condition_;
    //DISALLOW_COPY_AND_ASSGIN(FeatureCache);
};

}  // namespace ppspeech
