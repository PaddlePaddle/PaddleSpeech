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

class FeatureExtractorInterface {
  public:
    // accept input data, accept feature or raw waves which decided 
    // by the base_extractor
    virtual void Accept(
        const kaldi::VectorBase<kaldi::BaseFloat>& inputs) = 0;
    // get the processed result
    // the length of output = feature_row * feature_dim,
    // the Matrix is squashed into Vector
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* outputs) = 0;
    // the Dim is the feature dim
    virtual size_t Dim() const = 0;
    virtual void SetFinished() = 0;
    virtual bool IsFinished() const = 0;
    // virtual void Reset();
};

}  // namespace ppspeech
