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

// waves cache
class AudioCache : public FrontendInterface {
  public:
    explicit AudioCache(int buffer_size = 1000 * kint16max,
                        bool to_float32 = false);

    virtual void Accept(const std::vector<BaseFloat>& waves);

    virtual bool Read(std::vector<kaldi::BaseFloat>* waves);

    // the audio dim is 1, one sample, which is useless,
    // so we return size_(cache samples) instead.
    virtual size_t Dim() const { return size_; }

    virtual void SetFinished() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
    }

    virtual bool IsFinished() const { return finished_; }

    void Reset() override {
        offset_ = 0;
        size_ = 0;
        finished_ = false;
        nsamples_ = 0;
    }

  private:
    kaldi::BaseFloat Convert2PCM32(kaldi::BaseFloat val);

    std::vector<kaldi::BaseFloat> ring_buffer_;
    size_t offset_;    // offset in ring_buffer_, begin of data
    size_t size_;      // samples in ring_buffer_, size of valid data
    size_t capacity_;  // capacity of ring_buffer_, full size of data buffer,
                       // unit: sample
    bool finished_;    // reach audio end
    std::mutex mutex_;
    std::condition_variable ready_feed_condition_;
    kaldi::int32 timeout_;  // millisecond
    bool to_float32_;       // int16 -> float32. used in linear_spectrogram

    int32 nsamples_;  // number samples readed.
    DISALLOW_COPY_AND_ASSIGN(AudioCache);
};

}  // namespace ppspeech
