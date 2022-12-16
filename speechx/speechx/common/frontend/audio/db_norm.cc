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


#include "frontend/audio/db_norm.h"

#include "kaldi/feat/cmvn.h"
#include "kaldi/util/kaldi-io.h"

namespace ppspeech {

using kaldi::BaseFloat;
using kaldi::SubVector;
using kaldi::Vector;
using kaldi::VectorBase;
using std::unique_ptr;
using std::vector;

DecibelNormalizer::DecibelNormalizer(
    const DecibelNormalizerOptions& opts,
    std::unique_ptr<FrontendInterface> base_extractor) {
    base_extractor_ = std::move(base_extractor);
    opts_ = opts;
    dim_ = 1;
}

void DecibelNormalizer::Accept(const kaldi::VectorBase<BaseFloat>& waves) {
    base_extractor_->Accept(waves);
}

bool DecibelNormalizer::Read(kaldi::Vector<BaseFloat>* waves) {
    if (base_extractor_->Read(waves) == false || waves->Dim() == 0) {
        return false;
    }
    Compute(waves);
    return true;
}

bool DecibelNormalizer::Compute(VectorBase<BaseFloat>* waves) const {
    // calculate db rms
    BaseFloat rms_db = 0.0;
    BaseFloat mean_square = 0.0;
    BaseFloat gain = 0.0;
    BaseFloat wave_float_normlization = 1.0f / (std::pow(2, 16 - 1));

    vector<BaseFloat> samples;
    samples.resize(waves->Dim());
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = (*waves)(i);
    }

    // square
    for (auto& d : samples) {
        if (opts_.convert_int_float) {
            d = d * wave_float_normlization;
        }
        mean_square += d * d;
    }

    // mean
    mean_square /= samples.size();
    rms_db = 10 * std::log10(mean_square);
    gain = opts_.target_db - rms_db;

    if (gain > opts_.max_gain_db) {
        LOG(ERROR)
            << "Unable to normalize segment to " << opts_.target_db << "dB,"
            << "because the the probable gain have exceeds opts_.max_gain_db"
            << opts_.max_gain_db << "dB.";
        return false;
    }

    // Note that this is an in-place transformation.
    for (auto& item : samples) {
        // python item *= 10.0 ** (gain / 20.0)
        item *= std::pow(10.0, gain / 20.0);
    }

    std::memcpy(
        waves->Data(), samples.data(), sizeof(BaseFloat) * samples.size());
    return true;
}


}  // namespace ppspeech
