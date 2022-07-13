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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

#include "paddlespeech/audio/src/pybind/kaldi/kaldi_feature_wrapper.h"
#include "feat/pitch-functions.h"

namespace py = pybind11;

namespace paddleaudio {
namespace kaldi {

struct FbankOptions{
  bool use_energy;  // append an extra dimension with energy to the filter banks
  float energy_floor;
  bool raw_energy;  // If true, compute energy before preemphasis and windowing
  bool htk_compat;  // If true, put energy last (if using energy)
  bool use_log_fbank;  // if true (default), produce log-filterbank, else linear
  bool use_power; 
  FbankOptions(): use_energy(false),
                 energy_floor(0.0),
                 raw_energy(true),
                 htk_compat(false),
                 use_log_fbank(true),
                 use_power(true) {}
};

bool InitFbank(
    ::kaldi::FrameExtractionOptions frame_opts,
    ::kaldi::MelBanksOptions mel_opts,
    FbankOptions fbank_opts);

py::array_t<float> ComputeFbank(
    ::kaldi::FrameExtractionOptions frame_opts,
    ::kaldi::MelBanksOptions mel_opts,
    FbankOptions fbank_opts,
    const py::array_t<float>& wav);

py::array_t<float> ComputeFbankStreaming(const py::array_t<float>& wav);

void ResetFbank();

py::array_t<float> ComputeKaldiPitch(
    const ::kaldi::PitchExtractionOptions& opts,
    const py::array_t<float>& wav);

}  // namespace kaldi
}  // namespace paddleaudio
