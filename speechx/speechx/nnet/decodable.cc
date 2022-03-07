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

#include "nnet/decodable.h"

namespace ppspeech {

using kaldi::BaseFloat;
using kaldi::Matrix;

Decodable::Decodable(const std::shared_ptr<NnetInterface>& nnet)
    : frontend_(NULL), nnet_(nnet), finished_(false), frames_ready_(0) {}

void Decodable::Acceptlikelihood(const Matrix<BaseFloat>& likelihood) {
    frames_ready_ += likelihood.NumRows();
}

// Decodable::Init(DecodableConfig config) {
//}

bool Decodable::IsLastFrame(int32 frame) const {
    CHECK_LE(frame, frames_ready_);
    return finished_ && (frame == frames_ready_ - 1);
}

int32 Decodable::NumIndices() const { return 0; }

BaseFloat Decodable::LogLikelihood(int32 frame, int32 index) { return 0; }

void Decodable::FeedFeatures(const Matrix<kaldi::BaseFloat>& features) {
    nnet_->FeedForward(features, &nnet_cache_);
    frames_ready_ += nnet_cache_.NumRows();
    return;
}

std::vector<BaseFloat> Decodable::FrameLogLikelihood(int32 frame) {
    std::vector<BaseFloat> result;
    result.reserve(nnet_cache_.NumCols());
    for (int32 idx = 0; idx < nnet_cache_.NumCols(); ++idx) {
        result[idx] = nnet_cache_(frame, idx);
    }
    return result;
}

void Decodable::Reset() {
    // frontend_.Reset();
    nnet_->Reset();
}

}  // namespace ppspeech