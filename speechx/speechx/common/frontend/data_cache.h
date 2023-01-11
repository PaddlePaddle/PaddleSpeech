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
#include "frontend/frontend_itf.h"

using std::vector;

namespace ppspeech {

// Simulates audio/feature input, by returning data from a Vector.
// This class is mostly meant to be used for online decoder testing using
// pre-recorded audio/feature
class DataCache : public FrontendInterface {
  public:
    DataCache() : finished_{false}, dim_{0} {}

    // accept waves/feats
    void Accept(const std::vector<kaldi::BaseFloat>& inputs) override {
        data_ = std::move(inputs);
    }

    bool Read(vector<kaldi::BaseFloat>* feats) override {
        if (data_.size() == 0) {
            return false;
        }
        (*feats) = std::move(data_);
        data_.resize(0);
        return true;
    }

    void SetFinished() override { finished_ = true; }
    bool IsFinished() const override { return finished_; }
    size_t Dim() const override { return dim_; }
    void SetDim(int32 dim) { dim_ = dim; }
    void Reset() override {
        finished_ = true;
        dim_ = 0;
    }

  private:
    std::vector<kaldi::BaseFloat> data_;
    bool finished_;
    int32 dim_;

    DISALLOW_COPY_AND_ASSIGN(DataCache);
};
}  // namespace ppspeech