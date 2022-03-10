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

class RawAudioSource : public FeatureExtractorInterface {
  public:
    explicit RawAudioSource(int buffer_size = kint16max);
    virtual void AcceptWaveform(const kaldi::VectorBase<BaseFloat>& data);
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feat);
    virtual size_t Dim() const { return data_length_; }
    virtual void SetFinished() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
    }
    virtual bool IsFinished() const { return finished_; }

  private:
    std::vector<kaldi::BaseFloat> ring_buffer_;
    size_t start_;
    size_t data_length_;
    bool finished_;
    mutable std::mutex mutex_;
    std::condition_variable ready_feed_condition_;
    kaldi::int32 timeout_;

    DISALLOW_COPY_AND_ASSIGN(RawAudioSource);
};

// it is a datasource for testing different frontend module.
class RawDataSource : public FeatureExtractorInterface {
  public:
    explicit RawDataSource() { finished_ = false; }
    virtual void AcceptWaveform(
        const kaldi::VectorBase<kaldi::BaseFloat>& input) {
        data_ = input;
    }
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feat) {
        if (data_.Dim() == 0) {
            return false;
        }
        (*feat) = data_;
        data_.Resize(0);
        return true;
    }
    virtual size_t Dim() const { return data_.Dim(); }
    virtual void SetFinished() { finished_ = true; }
    virtual bool IsFinished() const { return finished_; }

  private:
    kaldi::Vector<kaldi::BaseFloat> data_;
    bool finished_;

    DISALLOW_COPY_AND_ASSIGN(RawDataSource);
};

}  // namespace ppspeech
