#include "frontend/feature_cache.h"

void FeatureCache::AcceptWaveform(const kaldi::VectorBase<kaldi::BaseFloat>& input) {
    base_extractor_->AcceptWaveform(input);
    // feed current data
    while (base_extractor_->IsLastFrame()) {
      Compute();
    }
}

// pop feature chunk 
void FeatureCache::Read(kaldi::VectorBase<kaldi::BaseFloat>* feat) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (cache_.empty()) {
        ready_read_condition_.wait(lock);
    }
    feat->CopyFromVec(cache_.front());
    cache_.pop();
    ready_feed_condition_.notify_one();
}

// read all data from base_feature_extractor_ into cache_
void FeatureCache::Compute() {
    // compute and feed
    Vector<BaseFloat> feature_chunk(base_extractor_->Dim());
    base_extractor_->Read(&feature_chunk);
    std::lock_guard<std::mutex> lock(mutex_);
    while (cache_.size() >= max_size_) {
        ready_feed_condition_.wait(lock);
    }
    cache_.push(feature_chunk);
    ready_read_condition_.notify_one();
}

// compute the last chunk data && set feed finished 
void FeatureCache::InputFinishd() {
    Compute();
}
