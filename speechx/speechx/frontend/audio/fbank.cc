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


#include "frontend/audio/fbank.h"
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

// todo refactor later:(SmileGoat)

Fbank::Fbank(const FbankOptions& opts,
             std::unique_ptr<FrontendInterface> base_extractor)
    : opts_(opts),
      computer_(opts.fbank_opts),
      window_function_(opts.fbank_opts.frame_opts) {
    base_extractor_ = std::move(base_extractor);
    chunk_sample_size_ = static_cast<int32>(
        opts.streaming_chunk * opts.fbank_opts.frame_opts.samp_freq);
}

void Fbank::Accept(const VectorBase<BaseFloat>& inputs) {
    base_extractor_->Accept(inputs);
}

bool Fbank::Read(Vector<BaseFloat>* feats) {
    Vector<BaseFloat> wav(chunk_sample_size_);
    bool flag = base_extractor_->Read(&wav);
    if (flag == false || wav.Dim() == 0) return false;

    // append remaned waves
    int32 wav_len = wav.Dim();
    int32 left_len = remained_wav_.Dim();
    Vector<BaseFloat> waves(left_len + wav_len);
    waves.Range(0, left_len).CopyFromVec(remained_wav_);
    waves.Range(left_len, wav_len).CopyFromVec(wav);

    // compute speech feature
    Compute(waves, feats);

    // cache remaned waves
    kaldi::FrameExtractionOptions frame_opts = computer_.GetFrameOptions();
    int32 num_frames = kaldi::NumFrames(waves.Dim(), frame_opts);
    int32 frame_shift = frame_opts.WindowShift();
    int32 left_samples = waves.Dim() - frame_shift * num_frames;
    remained_wav_.Resize(left_samples);
    remained_wav_.CopyFromVec(
        waves.Range(frame_shift * num_frames, left_samples));
    return true;
}

// Compute spectrogram feat
bool Fbank::Compute(const Vector<BaseFloat>& waves, Vector<BaseFloat>* feats) {
    const kaldi::FrameExtractionOptions& frame_opts =
        computer_.GetFrameOptions();
    int32 num_samples = waves.Dim();
    int32 frame_length = frame_opts.WindowSize();
    int32 sample_rate = frame_opts.samp_freq;
    if (num_samples < frame_length) {
        return true;
    }

    int32 num_frames = kaldi::NumFrames(num_samples, frame_opts);
    feats->Resize(num_frames * Dim());

    Vector<BaseFloat> window;
    bool need_raw_log_energy = computer_.NeedRawLogEnergy();
    for (int32 frame = 0; frame < num_frames; frame++) {
        BaseFloat raw_log_energy = 0.0;
        kaldi::ExtractWindow(0,
                             waves,
                             frame,
                             frame_opts,
                             window_function_,
                             &window,
                             need_raw_log_energy ? &raw_log_energy : NULL);


        Vector<BaseFloat> this_feature(computer_.Dim(), kaldi::kUndefined);
        // note: this online feature-extraction code does not support VTLN.
        RealFft(&window, true);
        kaldi::ComputePowerSpectrum(&window);
        const kaldi::MelBanks &mel_bank = *(computer_.GetMelBanks(1.0));
        SubVector<BaseFloat> power_spectrum(window, 0, window.Dim() / 2 + 1); 
        if (!opts_.fbank_opts.use_power) {
            power_spectrum.ApplyPow(0.5);
        }
        int32 mel_offset = ((opts_.fbank_opts.use_energy && !opts_.fbank_opts.htk_compat) ? 1 : 0);
        SubVector<BaseFloat> mel_energies(this_feature, mel_offset, opts_.fbank_opts.mel_opts.num_bins);
        mel_bank.Compute(power_spectrum, &mel_energies);
        mel_energies.ApplyFloor(1e-07);
        mel_energies.ApplyLog();
        SubVector<BaseFloat> output_row(feats->Data() + frame * Dim(), Dim());
        output_row.CopyFromVec(this_feature);
    }
    return true;
}

}  // namespace ppspeech
