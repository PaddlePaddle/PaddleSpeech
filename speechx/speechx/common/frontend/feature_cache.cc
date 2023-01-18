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

#include "frontend/feature_cache.h"

namespace ppspeech {

using kaldi::BaseFloat;
using std::unique_ptr;
using std::vector;

FeatureCache::FeatureCache(size_t max_size,
                           unique_ptr<FrontendInterface> base_extractor) {
    max_size_ = max_size;
    base_extractor_ = std::move(base_extractor);
    dim_ = base_extractor_->Dim();
}

void FeatureCache::Accept(const std::vector<kaldi::BaseFloat>& inputs) {
    // read inputs
    base_extractor_->Accept(inputs);
}

// pop feature chunk
bool FeatureCache::Read(std::vector<kaldi::BaseFloat>* feats) {
    kaldi::Timer timer;
    std::unique_lock<std::mutex> lock(mutex_);
    // feed current data
    if (cache_.empty()) {
        bool result = false;
        do {
            result = Compute();
        } while (result);
    }
    if (cache_.empty()) return false;

    // read from cache
    *feats = cache_.front();
    cache_.pop();
    //ready_feed_condition_.notify_one();
    VLOG(2) << "FeatureCache::Read cost: " << timer.Elapsed() << " sec.";
    return true;
}

// read all data from base_feature_extractor_ into cache_
bool FeatureCache::Compute() {
    // compute and feed
    vector<BaseFloat> feature;
    bool result = base_extractor_->Read(&feature);
    if (result == false || feature.size() == 0) return false;

    kaldi::Timer timer;

    int32 num_chunk = feature.size() / dim_;
    VLOG(3) << "nframe computed: " << nframe_;

    for (int chunk_idx = 0; chunk_idx < num_chunk; ++chunk_idx) {
        int32 start = chunk_idx * dim_;
        vector<BaseFloat> feature_chunk(feature.data() + start, 
                                        feature.data() + start + dim_);
        // feed cache
        cache_.push(feature_chunk);
        ++nframe_;
    }

    VLOG(1) << "FeatureCache::Compute cost: " << timer.Elapsed() << " sec. "
            << num_chunk << " feats.";
    return true;
}

}  // namespace ppspeech