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


#include "frontend/cmvn.h"

#include "utils/file_utils.h"
#include "utils/picojson.h"

namespace ppspeech {

using kaldi::BaseFloat;
using std::unique_ptr;
using std::vector;


CMVN::CMVN(std::string cmvn_file, unique_ptr<FrontendInterface> base_extractor)
    : var_norm_(true) {
    CHECK_NE(cmvn_file, "");
    base_extractor_ = std::move(base_extractor);
    ReadCMVNFromJson(cmvn_file);
    dim_ = mean_stats_.size() - 1;
}

void CMVN::ReadCMVNFromJson(std::string cmvn_file) {
    std::string json_str = ppspeech::ReadFile2String(cmvn_file);
    picojson::value value;
    std::string err;
    const char* json_end = picojson::parse(
        value, json_str.c_str(), json_str.c_str() + json_str.size(), &err);
    if (!value.is<picojson::object>()) {
        LOG(ERROR) << "Input json file format error.";
    }
    const picojson::value::array& mean_stat =
        value.get("mean_stat").get<picojson::array>();
    for (auto it = mean_stat.begin(); it != mean_stat.end(); it++) {
        mean_stats_.push_back((*it).get<double>());
    }

    const picojson::value::array& var_stat =
        value.get("var_stat").get<picojson::array>();
    for (auto it = var_stat.begin(); it != var_stat.end(); it++) {
        var_stats_.push_back((*it).get<double>());
    }

    kaldi::int32 frame_num = value.get("frame_num").get<int64_t>();
    LOG(INFO) << "nframe: " << frame_num;
    mean_stats_.push_back(frame_num);
    var_stats_.push_back(0);
}

void CMVN::Accept(const std::vector<kaldi::BaseFloat>& inputs) {
    // feed waves/feats to compute feature
    base_extractor_->Accept(inputs);
    return;
}

bool CMVN::Read(std::vector<BaseFloat>* feats) {
    // compute feature
    if (base_extractor_->Read(feats) == false || feats->size() == 0) {
        return false;
    }

    // appply cmvn
    kaldi::Timer timer;
    Compute(feats);
    VLOG(1) << "CMVN::Read cost: " << timer.Elapsed() << " sec.";
    return true;
}

// feats contain num_frames feature.
void CMVN::Compute(vector<BaseFloat>* feats) const {
    KALDI_ASSERT(feats != NULL);

    if (feats->size() % dim_ != 0) {
        LOG(ERROR) << "Dim mismatch: cmvn " << mean_stats_.size() << ','
                   << var_stats_.size() - 1 << ", feats " << feats->size()
                   << 'x';
    }
    if (var_stats_.size() == 0 && var_norm_) {
        LOG(ERROR)
            << "You requested variance normalization but no variance stats_ "
            << "are supplied.";
    }

    double count = mean_stats_[dim_];
    // Do not change the threshold of 1.0 here: in the balanced-cmvn code, when
    // computing an offset and representing it as stats_, we use a count of one.
    if (count < 1.0)
        LOG(ERROR) << "Insufficient stats_ for cepstral mean and variance "
                      "normalization: "
                   << "count = " << count;

    if (!var_norm_) {
        vector<BaseFloat> offset(feats->size());
        vector<double> mean_stats(mean_stats_);
        for (size_t i = 0; i < mean_stats.size(); ++i) {
            mean_stats[i] /= count;
        }
        vector<double> mean_stats_apply(feats->size());
        // fill the datat of mean_stats in mean_stats_appy whose dim_ is equal
        // with the dim_ of feature.
        // the dim_ of feats = dim_ * num_frames;
        for (int32 idx = 0; idx < feats->size() / dim_; ++idx) {
            std::memcpy(mean_stats_apply.data() + dim_ * idx,
                        mean_stats.data(),
                        dim_ * sizeof(double));
        }
        for (size_t idx = 0; idx < feats->size(); ++idx) {
            feats->at(idx) += offset[idx];
        }
        return;
    }
    // norm(0, d) = mean offset;
    // norm(1, d) = scale, e.g. x(d) <-- x(d)*norm(1, d) + norm(0, d).
    vector<BaseFloat> norm0(feats->size());
    vector<BaseFloat> norm1(feats->size());
    for (int32 d = 0; d < dim_; d++) {
        double mean, offset, scale;
        mean = mean_stats_[d] / count;
        double var = (var_stats_[d] / count) - mean * mean, floor = 1.0e-20;
        if (var < floor) {
            LOG(WARNING) << "Flooring cepstral variance from " << var << " to "
                         << floor;
            var = floor;
        }
        scale = 1.0 / sqrt(var);
        if (scale != scale || 1 / scale == 0.0)
            LOG(ERROR)
                << "NaN or infinity in cepstral mean/variance computation";
        offset = -(mean * scale);
        for (int32 d_skip = d; d_skip < feats->size();) {
            norm0[d_skip] = offset;
            norm1[d_skip] = scale;
            d_skip = d_skip + dim_;
        }
    }
    // Apply the normalization.
    for (size_t idx = 0; idx < feats->size(); ++idx) {
        feats->at(idx) *= norm1[idx];
    }

    for (size_t idx = 0; idx < feats->size(); ++idx) {
        feats->at(idx) += norm0[idx];
    }
}

}  // namespace ppspeech
