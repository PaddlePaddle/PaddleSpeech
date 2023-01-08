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

#include "paddleaudio/third_party/kaldi-native-fbank/csrc/feature-fbank.h"
#include "paddleaudio/src/pybind/kaldi/feature_common.h"

namespace paddleaudio {
namespace kaldi {

typedef StreamingFeatureTpl<knf::FbankComputer> Fbank;

class KaldiFeatureWrapper {
  public:
    static KaldiFeatureWrapper* GetInstance();
    bool InitFbank(knf::FbankOptions opts);
    py::array_t<float> ComputeFbank(const py::array_t<float> wav);
    int Dim() { return fbank_->Dim(); }
    void ResetFbank() { fbank_->Reset(); }

  private:
    std::unique_ptr<paddleaudio::kaldi::Fbank> fbank_;
};

}  // namespace kaldi
}  // namespace paddleaudio
