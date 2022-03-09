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


#include "frontend/normalizer.h"
#include "kaldi/feat/cmvn.h"
#include "kaldi/util/kaldi-io.h"

namespace ppspeech {

using kaldi::Vector;
using kaldi::VectorBase;
using kaldi::BaseFloat;
using std::vector;
using kaldi::SubVector;
using std::unique_ptr;

DecibelNormalizer::DecibelNormalizer(
    const DecibelNormalizerOptions& opts,
    std::unique_ptr<FeatureExtractorInterface> base_extractor) {
    base_extractor_ = std::move(base_extractor);
    opts_ = opts;
    dim_ = 0;
}

void DecibelNormalizer::AcceptWaveform(
    const kaldi::VectorBase<BaseFloat>& input) {
    // dim_ = input.Dim();
    // waveform_.Resize(input.Dim());
    // waveform_.CopyFromVec(input);
    base_extractor_->AcceptWaveform(input);
}

bool DecibelNormalizer::Read(kaldi::Vector<BaseFloat>* feat) {
    // if (waveform_.Dim() == 0) return;
    if (base_extractor_->Read(feat) == false || feat->Dim() == 0) {
        return false;
    }
    Compute(feat);
    return true;
}

// todo remove later
void CopyVector2StdVector(const kaldi::VectorBase<BaseFloat>& input,
                          vector<BaseFloat>* output) {
    if (input.Dim() == 0) return;
    output->resize(input.Dim());
    for (size_t idx = 0; idx < input.Dim(); ++idx) {
        (*output)[idx] = input(idx);
    }
}

void CopyStdVector2Vector(const vector<BaseFloat>& input,
                          VectorBase<BaseFloat>* output) {
    if (input.empty()) return;
    assert(input.size() == output->Dim());
    for (size_t idx = 0; idx < input.size(); ++idx) {
        (*output)(idx) = input[idx];
    }
}

bool DecibelNormalizer::Compute(VectorBase<BaseFloat>* feat) const {
    // calculate db rms
    BaseFloat rms_db = 0.0;
    BaseFloat mean_square = 0.0;
    BaseFloat gain = 0.0;
    BaseFloat wave_float_normlization = 1.0f / (std::pow(2, 16 - 1));

    vector<BaseFloat> samples;
    samples.resize(feat->Dim());
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = (*feat)(i);
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

    CopyStdVector2Vector(samples, feat);
    return true;
}

CMVN::CMVN(std::string cmvn_file,
           unique_ptr<FeatureExtractorInterface> base_extractor)
    : var_norm_(true) {
    base_extractor_ = std::move(base_extractor);
    bool binary;
    kaldi::Input ki(cmvn_file, &binary);
    stats_.Read(ki.Stream(), binary);
    dim_ = stats_.NumCols() - 1;
}

void CMVN::AcceptWaveform(const kaldi::VectorBase<kaldi::BaseFloat>& input) {
    base_extractor_->AcceptWaveform(input);
    return;
}

bool CMVN::Read(kaldi::Vector<BaseFloat>* feat) {
    if (base_extractor_->Read(feat) == false) {
        return false;
    }
    Compute(feat);
    return true;
}

// feats contain num_frames feature.
void CMVN::Compute(VectorBase<BaseFloat>* feats) const {
    KALDI_ASSERT(feats != NULL);
    int32 dim = stats_.NumCols() - 1;
    if (stats_.NumRows() > 2 || stats_.NumRows() < 1 ||
        feats->Dim() % dim != 0) {
        KALDI_ERR << "Dim mismatch: cmvn " << stats_.NumRows() << 'x'
                  << stats_.NumCols() << ", feats " << feats->Dim() << 'x';
    }
    if (stats_.NumRows() == 1 && var_norm_) {
        KALDI_ERR
            << "You requested variance normalization but no variance stats_ "
            << "are supplied.";
    }

    double count = stats_(0, dim);
    // Do not change the threshold of 1.0 here: in the balanced-cmvn code, when
    // computing an offset and representing it as stats_, we use a count of one.
    if (count < 1.0)
        KALDI_ERR << "Insufficient stats_ for cepstral mean and variance "
                     "normalization: "
                  << "count = " << count;

    if (!var_norm_) {
        Vector<BaseFloat> offset(feats->Dim());
        SubVector<double> mean_stats(stats_.RowData(0), dim);
        Vector<double> mean_stats_apply(feats->Dim());
        // fill the datat of mean_stats in mean_stats_appy whose dim is equal
        // with the dim of feature.
        // the dim of feats = dim * num_frames;
        for (int32 idx = 0; idx < feats->Dim() / dim; ++idx) {
            SubVector<double> stats_tmp(mean_stats_apply.Data() + dim * idx,
                                        dim);
            stats_tmp.CopyFromVec(mean_stats);
        }
        offset.AddVec(-1.0 / count, mean_stats_apply);
        feats->AddVec(1.0, offset);
        return;
    }
    // norm(0, d) = mean offset;
    // norm(1, d) = scale, e.g. x(d) <-- x(d)*norm(1, d) + norm(0, d).
    kaldi::Matrix<BaseFloat> norm(2, feats->Dim());
    for (int32 d = 0; d < dim; d++) {
        double mean, offset, scale;
        mean = stats_(0, d) / count;
        double var = (stats_(1, d) / count) - mean * mean, floor = 1.0e-20;
        if (var < floor) {
            KALDI_WARN << "Flooring cepstral variance from " << var << " to "
                       << floor;
            var = floor;
        }
        scale = 1.0 / sqrt(var);
        if (scale != scale || 1 / scale == 0.0)
            KALDI_ERR
                << "NaN or infinity in cepstral mean/variance computation";
        offset = -(mean * scale);
        for (int32 d_skip = d; d_skip < feats->Dim();) {
            norm(0, d_skip) = offset;
            norm(1, d_skip) = scale;
            d_skip = d_skip + dim;
        }
    }
    // Apply the normalization.
    feats->MulElements(norm.Row(1));
    feats->AddVec(1.0, norm.Row(0));
}

void CMVN::ApplyCMVN(kaldi::MatrixBase<BaseFloat>* feats) {
    ApplyCmvn(stats_, var_norm_, feats);
}

}  // namespace ppspeech
