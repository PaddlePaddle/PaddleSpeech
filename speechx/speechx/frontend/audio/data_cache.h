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

// Simulates audio/feature input, by returning data from a Vector.
// This class is mostly meant to be used for online decoder testing using
// pre-recorded audio/feature
class DataCache : public FrontendInterface {
  public:
    explicit DataCache() { finished_ = false; }

    // accept waves/feats
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

    virtual void SetFinished() { finished_ = true; }
    virtual bool IsFinished() const { return finished_; }
    virtual size_t Dim() const { return dim_; }
    void SetDim(int32 dim) { dim_ = dim; }
    virtual void Reset() { finished_ = true; }

  private:
    kaldi::Vector<kaldi::BaseFloat> data_;
    bool finished_;
    int32 dim_;

    DISALLOW_COPY_AND_ASSIGN(DataCache);
};
}