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

namespace py = pybind11;

namespace paddleaudio {
namespace kaldi {

bool InitFbank(float samp_freq,  // frame opts
               float frame_shift_ms,
               float frame_length_ms,
               float dither,
               float preemph_coeff,
               bool remove_dc_offset,
               std::string window_type,  // e.g. Hamming window
               bool round_to_power_of_two,
               float blackman_coeff,
               bool snip_edges,
               bool allow_downsample,
               bool allow_upsample,
               int max_feature_vectors,
               int num_bins,  // mel opts
               float low_freq,
               float high_freq,
               float vtln_low,
               float vtln_high,
               bool debug_mel,
               bool htk_mode,
               bool use_energy,  // fbank opts
               float energy_floor,
               bool raw_energy,
               bool htk_compat,
               bool use_log_fbank,
               bool use_power);

py::array_t<double> ComputeFbank(
    float samp_freq,  // frame opts
    float frame_shift_ms,
    float frame_length_ms,
    float dither,
    float preemph_coeff,
    bool remove_dc_offset,
    std::string window_type,  // e.g. Hamming window
    bool round_to_power_of_two,
    ::kaldi::BaseFloat blackman_coeff,
    bool snip_edges,
    bool allow_downsample,
    bool allow_upsample,
    int max_feature_vectors,
    int num_bins,  // mel opts
    float low_freq,
    float high_freq,
    float vtln_low,
    float vtln_high,
    bool debug_mel,
    bool htk_mode,
    bool use_energy,  // fbank opts
    float energy_floor,
    bool raw_energy,
    bool htk_compat,
    bool use_log_fbank,
    bool use_power,
    const py::array_t<double>& wav);

py::array_t<double> ComputeFbankStreaming(const py::array_t<double>& wav);

void ResetFbank();

py::array_t<double> ComputeFbankStreaming(const py::array_t<double>& wav);

py::array_t<double> TestFun(const py::array_t<double>& wav);

}  // namespace kaldi
}  // namespace paddleaudio
