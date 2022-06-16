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

FbankComputer::FbankComputer(const Options& opts)
    : opts_(opts), computer_(opts) {}

int32 FbankComputer::Dim() const {
    return opts_.mel_opts.num_bins + (opts_.use_energy ? 1 : 0);
}

bool FbankComputer::NeedRawLogEnergy() {
    return opts_.use_energy && opts_.raw_energy;
}

// Compute feat
bool FbankComputer::Compute(Vector<BaseFloat>* window,
                            Vector<BaseFloat>* feat) {
    RealFft(window, true);
    kaldi::ComputePowerSpectrum(window);
    const kaldi::MelBanks& mel_bank = *(computer_.GetMelBanks(1.0));
    SubVector<BaseFloat> power_spectrum(*window, 0, window->Dim() / 2 + 1);
    if (!opts_.use_power) {
        power_spectrum.ApplyPow(0.5);
    }
    int32 mel_offset = ((opts_.use_energy && !opts_.htk_compat) ? 1 : 0);
    SubVector<BaseFloat> mel_energies(
        *feat, mel_offset, opts_.mel_opts.num_bins);
    mel_bank.Compute(power_spectrum, &mel_energies);
    mel_energies.ApplyFloor(1e-07);
    mel_energies.ApplyLog();
    return true;
}

}  // namespace ppspeech
