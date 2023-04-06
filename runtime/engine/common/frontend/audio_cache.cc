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

#include "frontend/audio_cache.h"

#include "kaldi/base/timer.h"

namespace ppspeech {

using kaldi::BaseFloat;
using std::vector;

AudioCache::AudioCache(int buffer_size, bool to_float32)
    : finished_(false),
      capacity_(buffer_size),  // unit: sample
      size_(0),
      offset_(0),
      timeout_(1),  // ms
      to_float32_(to_float32) {
    ring_buffer_.resize(capacity_);
}

BaseFloat AudioCache::Convert2PCM32(BaseFloat val) {
    // sample type int16, int16->float32
    return val * (1. / std::pow(2.0, 15));
}

void AudioCache::Accept(const vector<BaseFloat>& waves) {
    kaldi::Timer timer;
    std::unique_lock<std::mutex> lock(mutex_);
    while (size_ + waves.size() > ring_buffer_.size()) {
        ready_feed_condition_.wait(lock);
    }
    for (size_t idx = 0; idx < waves.size(); ++idx) {
        int32 buffer_idx = (idx + offset_ + size_) % ring_buffer_.size();
        ring_buffer_[buffer_idx] = waves[idx];
        if (to_float32_) ring_buffer_[buffer_idx] = Convert2PCM32(waves[idx]);
    }
    size_ += waves.size();
    VLOG(1) << "AudioCache::Accept cost: " << timer.Elapsed() << " sec. "
            << waves.size() << " samples.";
}

bool AudioCache::Read(vector<BaseFloat>* waves) {
    kaldi::Timer timer;
    size_t chunk_size = waves->size();
    std::unique_lock<std::mutex> lock(mutex_);
    if (chunk_size > size_) {
        if (finished_ == false) {
            return false;
        } else {
            // read last chunk data
            chunk_size = size_;
            waves->resize(chunk_size);
        }
    }

    for (size_t idx = 0; idx < chunk_size; ++idx) {
        int buff_idx = (offset_ + idx) % ring_buffer_.size();
        waves->at(idx) = ring_buffer_[buff_idx];
    }
    size_ -= chunk_size;
    offset_ = (offset_ + chunk_size) % ring_buffer_.size();

    nsamples_ += chunk_size;
    VLOG(3) << "nsamples readed: " << nsamples_;

    ready_feed_condition_.notify_one();
    VLOG(1) << "AudioCache::Read cost: " << timer.Elapsed() << " sec. "
            << chunk_size << " samples.";
    return true;
}

}  // namespace ppspeech
