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

#include "frontend/audio/feature_cache.h"

namespace ppspeech {

using kaldi::Vector;
using kaldi::VectorBase;
using kaldi::BaseFloat;
using std::vector;
using kaldi::SubVector;
using std::unique_ptr;

FeatureCache::FeatureCache(int max_size,
                           unique_ptr<FrontendInterface> base_extractor) {
    max_size_ = max_size;
    base_extractor_ = std::move(base_extractor);
}

void FeatureCache::Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs) {
    base_extractor_->Accept(inputs);
    // feed current data
    bool result = false;
    do {
        result = Compute();
    } while (result);
}

// pop feature chunk
bool FeatureCache::Read(kaldi::Vector<kaldi::BaseFloat>* feats) {
    kaldi::Timer timer;

    std::unique_lock<std::mutex> lock(mutex_);
    while (cache_.empty() && base_extractor_->IsFinished() == false) {
        ready_read_condition_.wait(lock);
        BaseFloat elapsed = timer.Elapsed() * 1000;
        // todo replace 1.0 with timeout_
        if (elapsed > 1.0) {
            return false;
        }
        usleep(1000);  // sleep 1 ms
    }
    if (cache_.empty()) return false;
    feats->Resize(cache_.front().Dim());
    feats->CopyFromVec(cache_.front());
    cache_.pop();
    ready_feed_condition_.notify_one();
    return true;
}

// read all data from base_feature_extractor_ into cache_
bool FeatureCache::Compute() {
    // compute and feed
    Vector<BaseFloat> feature_chunk;
    bool result = base_extractor_->Read(&feature_chunk);

    std::unique_lock<std::mutex> lock(mutex_);
    while (cache_.size() >= max_size_) {
        ready_feed_condition_.wait(lock);
    }

    // feed cache
    if (feature_chunk.Dim() != 0) {
        cache_.push(feature_chunk);
    }
    ready_read_condition_.notify_one();
    return result;
}

void Reset() {
    // std::lock_guard<std::mutex> lock(mutex_);
    return;
}

}  // namespace ppspeech