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

#include "paddlespeech/audio/src/pybind/kaldi/kaldi_feature.h"
#include "feat/pitch-functions.h"

namespace paddleaudio {
namespace kaldi {

bool InitFbank(
    ::kaldi::FrameExtractionOptions frame_opts,
    ::kaldi::MelBanksOptions mel_opts,
    FbankOptions fbank_opts) {
    ::kaldi::FbankOptions opts;
    opts.frame_opts = frame_opts;
    opts.mel_opts = mel_opts;
    opts.use_energy = fbank_opts.use_energy;
    opts.energy_floor = fbank_opts.energy_floor;
    opts.raw_energy = fbank_opts.raw_energy;
    opts.htk_compat = fbank_opts.htk_compat;
    opts.use_log_fbank = fbank_opts.use_log_fbank;
    opts.use_power = fbank_opts.use_power;
    paddleaudio::kaldi::KaldiFeatureWrapper::GetInstance()->InitFbank(opts);
    return true;
}

py::array_t<float> ComputeFbankStreaming(const py::array_t<float>& wav) {
    return paddleaudio::kaldi::KaldiFeatureWrapper::GetInstance()->ComputeFbank(
        wav);
}

py::array_t<float> ComputeFbank(
    ::kaldi::FrameExtractionOptions frame_opts,
    ::kaldi::MelBanksOptions mel_opts,
    FbankOptions fbank_opts,
    const py::array_t<float>& wav) {
    InitFbank(frame_opts, mel_opts, fbank_opts);
    py::array_t<float> result = ComputeFbankStreaming(wav);
    paddleaudio::kaldi::KaldiFeatureWrapper::GetInstance()->ResetFbank();
    return result;
}

void ResetFbank() {
    paddleaudio::kaldi::KaldiFeatureWrapper::GetInstance()->ResetFbank();
}

py::array_t<float> ComputeKaldiPitch(
  const ::kaldi::PitchExtractionOptions& opts,
  const py::array_t<float>& wav) {
    py::buffer_info info = wav.request();
    ::kaldi::SubVector<::kaldi::BaseFloat> input_wav((float*)info.ptr, info.size);
   
    ::kaldi::Matrix<::kaldi::BaseFloat> features;
    ::kaldi::ComputeKaldiPitch(opts, input_wav, &features);
    auto result = py::array_t<float>({features.NumRows(), features.NumCols()});
    for (int row_idx = 0; row_idx < features.NumRows(); ++row_idx) {
        std::memcpy(result.mutable_data(row_idx), features.Row(row_idx).Data(),
                    sizeof(float)*features.NumCols());
    }
   return result;
}

}  // namespace kaldi
}  // namespace paddleaudio
