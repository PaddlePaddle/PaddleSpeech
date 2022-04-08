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

#include "base/basic_types.h"
#include "kaldi/matrix/kaldi-vector.h"

namespace ppspeech {

class FrontendInterface {
  public:
    // Feed inputs: features(2D saved in 1D) or waveforms(1D).
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs) = 0;

    // Fetch processed data: features or waveforms.
    // For features(2D saved in 1D), the Matrix is squashed into Vector,
    //    the length of output = feature_row * feature_dim.
    // For waveforms(1D), samples saved in vector.
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* outputs) = 0;

    // Dim is the feature dim. For waveforms(1D), Dim is zero; else is specific,
    // e.g 80 for fbank.
    virtual size_t Dim() const = 0;

    // End Flag for Streaming Data.
    virtual void SetFinished() = 0;

    // whether is end of Streaming Data.
    virtual bool IsFinished() const = 0;

    // Reset to start state.
    virtual void Reset() = 0;
};

}  // namespace ppspeech
