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


#include "frontend/audio/cmvn.h"
#include "kaldi/feat/cmvn.h"
#include "kaldi/util/kaldi-io.h"

namespace ppspeech {

using kaldi::Vector;
using kaldi::VectorBase;
using kaldi::BaseFloat;
using std::vector;
using kaldi::SubVector;
using std::unique_ptr;


CMVN::CMVN(std::string cmvn_file, unique_ptr<FrontendInterface> base_extractor)
    : var_norm_(true) {
    base_extractor_ = std::move(base_extractor);
    bool binary;
    kaldi::Input ki(cmvn_file, &binary);
    stats_.Read(ki.Stream(), binary);
    dim_ = stats_.NumCols() - 1;
}

void CMVN::Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs) {
    // feed waves/feats to compute feature
    base_extractor_->Accept(inputs);
    return;
}

bool CMVN::Read(kaldi::Vector<BaseFloat>* feats) {
    // compute feature
    if (base_extractor_->Read(feats) == false || feats->Dim() == 0) {
        return false;
    }
    // appply cmvn
    Compute(feats);
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
