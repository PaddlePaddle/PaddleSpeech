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

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "kaldi-native-fbank/csrc/feature-window.h"

namespace paddleaudio {
namespace kaldi {

namespace py = pybind11;

template <class F>
class StreamingFeatureTpl {
  public:
    typedef typename F::Options Options;
    StreamingFeatureTpl(const Options& opts);
    bool ComputeFeature(const std::vector<float>& wav,
                        std::vector<float>* feats);
    void Reset() { remained_wav_.resize(0); }

    int Dim() { return computer_.Dim(); }

  private:
    bool Compute(const std::vector<float>& waves,
                 std::vector<float>* feats);
    Options opts_;
    knf::FeatureWindowFunction window_function_;
    std::vector<float> remained_wav_;
    F computer_;
};

}  // namespace kaldi
}  // namespace ppspeech

#include "feature_common_inl.h"
