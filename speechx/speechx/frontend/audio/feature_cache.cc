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

FeatureCache::FeatureCache(FeatureCacheOptions opts,
                           unique_ptr<FrontendInterface> base_extractor) {
    max_size_ = opts.max_size;
    frame_chunk_stride_ = opts.frame_chunk_stride;
    frame_chunk_size_ = opts.frame_chunk_size;
    base_extractor_ = std::move(base_extractor);
    dim_ = base_extractor_->Dim();
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
        // todo refactor: wait
        // ready_read_condition_.wait(lock);
        int32 elapsed = static_cast<int32>(timer.Elapsed() * 1000);
        // todo replace 1 with timeout_, 1 ms
        if (elapsed > 1) {
            return false;
        }
        usleep(100);  // sleep 0.1 ms
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
    Vector<BaseFloat> feature;
    bool result = base_extractor_->Read(&feature);
    if (result == false || feature.Dim() == 0) return false;
    int32 joint_len = feature.Dim() + remained_feature_.Dim();
    int32 num_chunk =
        ((joint_len / dim_) - frame_chunk_size_) / frame_chunk_stride_ + 1;

    Vector<BaseFloat> joint_feature(joint_len);
    joint_feature.Range(0, remained_feature_.Dim())
        .CopyFromVec(remained_feature_);
    joint_feature.Range(remained_feature_.Dim(), feature.Dim())
        .CopyFromVec(feature);

    for (int chunk_idx = 0; chunk_idx < num_chunk; ++chunk_idx) {
        int32 start = chunk_idx * frame_chunk_stride_ * dim_;
        Vector<BaseFloat> feature_chunk(frame_chunk_size_ * dim_);
        SubVector<BaseFloat> tmp(joint_feature.Data() + start,
                                 frame_chunk_size_ * dim_);
        feature_chunk.CopyFromVec(tmp);

        std::unique_lock<std::mutex> lock(mutex_);
        while (cache_.size() >= max_size_) {
            ready_feed_condition_.wait(lock);
        }

        // feed cache
        cache_.push(feature_chunk);
        ready_read_condition_.notify_one();
    }
    int32 remained_feature_len =
        joint_len - num_chunk * frame_chunk_stride_ * dim_;
    remained_feature_.Resize(remained_feature_len);
    remained_feature_.CopyFromVec(joint_feature.Range(
        frame_chunk_stride_ * num_chunk * dim_, remained_feature_len));
    return result;
}

}  // namespace ppspeech