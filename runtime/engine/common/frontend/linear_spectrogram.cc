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

#include "frontend/audio/linear_spectrogram.h"

#include "kaldi/base/kaldi-math.h"
#include "kaldi/feat/feature-common.h"
#include "kaldi/feat/feature-functions.h"
#include "kaldi/matrix/matrix-functions.h"

namespace ppspeech {

using kaldi::BaseFloat;
using kaldi::int32;
using kaldi::Matrix;
using kaldi::SubVector;
using kaldi::Vector;
using kaldi::VectorBase;
using std::vector;

LinearSpectrogramComputer::LinearSpectrogramComputer(const Options& opts)
    : opts_(opts) {
    kaldi::FeatureWindowFunction feature_window_function(opts.frame_opts);
    int32 window_size = opts.frame_opts.WindowSize();
    frame_length_ = window_size;
    dim_ = window_size / 2 + 1;
    BaseFloat hanning_window_energy = kaldi::VecVec(
        feature_window_function.window, feature_window_function.window);
    int32 sample_rate = opts.frame_opts.samp_freq;
    scale_ = 2.0 / (hanning_window_energy * sample_rate);
}

// Compute spectrogram feat
bool LinearSpectrogramComputer::Compute(Vector<BaseFloat>* window,
                                        Vector<BaseFloat>* feat) {
    window->Resize(frame_length_, kaldi::kCopyData);
    RealFft(window, true);
    kaldi::ComputePowerSpectrum(window);
    SubVector<BaseFloat> power_spectrum(*window, 0, dim_);
    power_spectrum.Scale(scale_);
    power_spectrum(0) = power_spectrum(0) / 2;
    power_spectrum(dim_ - 1) = power_spectrum(dim_ - 1) / 2;
    power_spectrum.Add(1e-14);
    power_spectrum.ApplyLog();
    feat->CopyFromVec(power_spectrum);
    return true;
}

}  // namespace ppspeech