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

#pragma once

namespace ppspeech {

class RawAudioCache : public FeatureExtractorInterface {
  public:
    explicit RawAudioCache(int buffer_size = kint16max);
    virtual void Accept(const kaldi::VectorBase<BaseFloat>& waves);
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* waves);
    // the audio dim is 1
    virtual size_t Dim() const { return 1; }
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

    DISALLOW_COPY_AND_ASSIGN(RawAudioCache);
};

// it is a datasource for testing different frontend module.
// it accepts waves or feats.
class RawDataCache : public FeatureExtractorInterface {
  public:
    explicit RawDataCache() { finished_ = false; }
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs) {
        data_ = inputs;
    }
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feats) {
        if (data_.Dim() == 0) {
            return false;
        }
        (*feats) = data_;
        data_.Resize(0);
        return true;
    }
    virtual size_t Dim() const { return dim_; }
    virtual void SetFinished() { finished_ = true; }
    virtual bool IsFinished() const { return finished_; }
    void SetDim(int32 dim) { dim_ = dim; }

  private:
    kaldi::Vector<kaldi::BaseFloat> data_;
    bool finished_;
    int32 dim_;

    DISALLOW_COPY_AND_ASSIGN(RawDataCache);
};

}  // namespace ppspeech
