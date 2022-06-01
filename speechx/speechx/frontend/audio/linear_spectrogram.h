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
#include "kaldi/feat/feature-window.h"
#include "frontend/audio/feature_common.h"

namespace ppspeech {

struct LinearSpectrogramOptions {
    kaldi::FrameExtractionOptions frame_opts;
    LinearSpectrogramOptions(): frame_opts() {}
};

class LinearSpectrogramComputer {
  public:
    typedef LinearSpectrogramOptions Options;
    explicit LinearSpectrogramComputer(const Options& opts);

    kaldi::FrameExtractionOptions& GetFrameOptions() {
      return opts_.frame_opts;
    }

    bool Compute(kaldi::Vector<kaldi::BaseFloat>* window,
                 kaldi::Vector<kaldi::BaseFloat>* feat);

    int32 Dim() const { return dim_; }

    bool NeedRawLogEnergy() { return false; }
    
  private:
    kaldi::BaseFloat scale_;
    Options opts_;
    int32 frame_length_;
    int32 dim_;
};

typedef StreamingFeatureTpl<LinearSpectrogramComputer> LinearSpectrogram;

    //DISALLOW_COPY_AND_ASSIGN(LinearSpectrogram);


}  // namespace ppspeech