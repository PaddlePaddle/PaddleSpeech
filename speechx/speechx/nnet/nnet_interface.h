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
#include "kaldi/base/kaldi-types.h"
#include "kaldi/matrix/kaldi-matrix.h"

namespace ppspeech {

class NnetInterface {
  public:
    virtual void FeedForward(const kaldi::Vector<kaldi::BaseFloat>& features,
                             int32 feature_dim,
                             kaldi::Vector<kaldi::BaseFloat>* inferences,
                             int32* inference_dim) = 0;
    virtual void Reset() = 0;
    virtual ~NnetInterface() {}
};

}  // namespace ppspeech
