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

using kaldi::int32;
using kaldi::BaseFloat;
using kaldi::Vector;
using kaldi::SubVector;
using kaldi::VectorBase;
using kaldi::Matrix;
using std::vector;

LinearSpectrogram::LinearSpectrogram(
    const LinearSpectrogramOptions& opts,
    std::unique_ptr<FrontendInterface> base_extractor)
    : opts_(opts), feature_window_funtion_(opts.frame_opts) {
    base_extractor_ = std::move(base_extractor);
    int32 window_size = opts.frame_opts.WindowSize();
    int32 window_shift = opts.frame_opts.WindowShift();
    dim_ = window_size / 2 + 1;
    chunk_sample_size_ =
        static_cast<int32>(opts.streaming_chunk * opts.frame_opts.samp_freq);
    hanning_window_energy_ = kaldi::VecVec(feature_window_funtion_.window,
                                           feature_window_funtion_.window);
}

void LinearSpectrogram::Accept(const VectorBase<BaseFloat>& inputs) {
    base_extractor_->Accept(inputs);
}

bool LinearSpectrogram::Read(Vector<BaseFloat>* feats) {
    Vector<BaseFloat> input_feats(chunk_sample_size_);
    bool flag = base_extractor_->Read(&input_feats);
    if (flag == false || input_feats.Dim() == 0) return false;

    int32 feat_len = input_feats.Dim();
    int32 left_len = remained_wav_.Dim();
    Vector<BaseFloat> waves(feat_len + left_len);
    waves.Range(0, left_len).CopyFromVec(remained_wav_);
    waves.Range(left_len, feat_len).CopyFromVec(input_feats);
    Compute(waves, feats);
    int32 frame_shift = opts_.frame_opts.WindowShift();
    int32 num_frames = kaldi::NumFrames(waves.Dim(), opts_.frame_opts);
    int32 left_samples = waves.Dim() - frame_shift * num_frames;
    remained_wav_.Resize(left_samples);
    remained_wav_.CopyFromVec(
        waves.Range(frame_shift * num_frames, left_samples));
    return true;
}

// Compute spectrogram feat
bool LinearSpectrogram::Compute(const Vector<BaseFloat>& waves,
                                Vector<BaseFloat>* feats) {
    int32 num_samples = waves.Dim();
    int32 frame_length = opts_.frame_opts.WindowSize();
    int32 sample_rate = opts_.frame_opts.samp_freq;
    BaseFloat scale = 2.0 / (hanning_window_energy_ * sample_rate);

    if (num_samples < frame_length) {
        return true;
    }

    int32 num_frames = kaldi::NumFrames(num_samples, opts_.frame_opts);
    feats->Resize(num_frames * dim_);
    Vector<BaseFloat> window;

    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        kaldi::ExtractWindow(0,
                             waves,
                             frame_idx,
                             opts_.frame_opts,
                             feature_window_funtion_,
                             &window,
                             NULL);

        SubVector<BaseFloat> output_row(feats->Data() + frame_idx * dim_, dim_);
        window.Resize(frame_length, kaldi::kCopyData);
        RealFft(&window, true);
        kaldi::ComputePowerSpectrum(&window);
        SubVector<BaseFloat> power_spectrum(window, 0, dim_);
        power_spectrum.Scale(scale);
        power_spectrum(0) = power_spectrum(0) / 2;
        power_spectrum(dim_ - 1) = power_spectrum(dim_ - 1) / 2;
        power_spectrum.Add(1e-14);
        power_spectrum.ApplyLog();
        output_row.CopyFromVec(power_spectrum);
    }
    return true;
}

}  // namespace ppspeech