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

#include "base/kaldi-common.h"

namespace paddleaudio {
namespace kaldi {  

template <class F>
StreamingFeatureTpl<F>::StreamingFeatureTpl(const Options& opts)
    : opts_(opts), computer_(opts), window_function_(opts.frame_opts) {
    // window_function_(computer_.GetFrameOptions()) { the opt set to zero
}

template <class F>
bool StreamingFeatureTpl<F>::ComputeFeature(
    const ::kaldi::VectorBase<::kaldi::BaseFloat>& wav,
    ::kaldi::Vector<::kaldi::BaseFloat>* feats) {
    // append remaned waves
    ::kaldi::int32 wav_len = wav.Dim();
    if (wav_len == 0) return false;
    ::kaldi::int32 left_len = remained_wav_.Dim();
    ::kaldi::Vector<::kaldi::BaseFloat> waves(left_len + wav_len);
    waves.Range(0, left_len).CopyFromVec(remained_wav_);
    waves.Range(left_len, wav_len).CopyFromVec(wav);

    // cache remaned waves
    ::kaldi::FrameExtractionOptions frame_opts = computer_.GetFrameOptions();
    ::kaldi::int32 num_frames = ::kaldi::NumFrames(waves.Dim(), frame_opts);
    ::kaldi::int32 frame_shift = frame_opts.WindowShift();
    ::kaldi::int32 left_samples = waves.Dim() - frame_shift * num_frames;
    remained_wav_.Resize(left_samples);
    remained_wav_.CopyFromVec(
        waves.Range(frame_shift * num_frames, left_samples));

    // compute speech feature
    Compute(waves, feats);
    return true;
}

// Compute feat
template <class F>
bool StreamingFeatureTpl<F>::Compute(
    const ::kaldi::Vector<::kaldi::BaseFloat>& waves,
    ::kaldi::Vector<::kaldi::BaseFloat>* feats) {
    ::kaldi::BaseFloat vtln_warp = 1.0;
    const ::kaldi::FrameExtractionOptions& frame_opts =
        computer_.GetFrameOptions();
    ::kaldi::int32 num_samples = waves.Dim();
    ::kaldi::int32 frame_length = frame_opts.WindowSize();
    ::kaldi::int32 sample_rate = frame_opts.samp_freq;
    if (num_samples < frame_length) {
        return false;
    }

    ::kaldi::int32 num_frames = ::kaldi::NumFrames(num_samples, frame_opts);
    feats->Resize(num_frames * Dim());

    ::kaldi::Vector<::kaldi::BaseFloat> window;
    bool need_raw_log_energy = computer_.NeedRawLogEnergy();
    for (::kaldi::int32 frame = 0; frame < num_frames; frame++) {
        ::kaldi::BaseFloat raw_log_energy = 0.0;
        ::kaldi::ExtractWindow(0,
                             waves,
                             frame,
                             frame_opts,
                             window_function_,
                             &window,
                             need_raw_log_energy ? &raw_log_energy : NULL);

        ::kaldi::Vector<::kaldi::BaseFloat> this_feature(computer_.Dim(),
                                                     ::kaldi::kUndefined);
        computer_.Compute(raw_log_energy, vtln_warp, &window, &this_feature);
        ::kaldi::SubVector<::kaldi::BaseFloat> output_row(
            feats->Data() + frame * Dim(), Dim());
        output_row.CopyFromVec(this_feature);
    }
    return true;
}

}  // namespace kaldi
}  // namespace paddleaudio
