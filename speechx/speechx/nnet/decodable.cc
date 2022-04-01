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
using std::vector;
using kaldi::Vector;

Decodable::Decodable(const std::shared_ptr<NnetInterface>& nnet,
                     const std::shared_ptr<FrontendInterface>& frontend)
    : frontend_(frontend), nnet_(nnet), frame_offset_(0), frames_ready_(0) {}

void Decodable::Acceptlikelihood(const Matrix<BaseFloat>& likelihood) {
    nnet_cache_ = likelihood;
    frames_ready_ += likelihood.NumRows();
}

// Decodable::Init(DecodableConfig config) {
//}

bool Decodable::IsLastFrame(int32 frame) const {
    CHECK_LE(frame, frames_ready_);
    return IsInputFinished() && (frame == frames_ready_ - 1);
}

int32 Decodable::NumIndices() const { return 0; }

BaseFloat Decodable::LogLikelihood(int32 frame, int32 index) {
    CHECK_LE(index, nnet_cache_.NumCols());
    return 0;
}

bool Decodable::EnsureFrameHaveComputed(int32 frame) {
    if (frame >= frames_ready_) {
        return AdvanceChunk();
    }
    return true;
}

bool Decodable::AdvanceChunk() {
    Vector<BaseFloat> features;
    if (frontend_ == NULL || frontend_->Read(&features) == false) {
        return false;
    }
    int32 nnet_dim = 0;
    Vector<BaseFloat> inferences;
    nnet_->FeedForward(features, frontend_->Dim(), &inferences, &nnet_dim);
    nnet_cache_.Resize(inferences.Dim() / nnet_dim, nnet_dim);
    nnet_cache_.CopyRowsFromVec(inferences);
    frame_offset_ = frames_ready_;
    frames_ready_ += nnet_cache_.NumRows();
    return true;
}

bool Decodable::FrameLogLikelihood(int32 frame, vector<BaseFloat>* likelihood) {
    std::vector<BaseFloat> result;
    if (EnsureFrameHaveComputed(frame) == false) return false;
    likelihood->resize(nnet_cache_.NumCols());
    for (int32 idx = 0; idx < nnet_cache_.NumCols(); ++idx) {
        (*likelihood)[idx] = nnet_cache_(frame - frame_offset_, idx);
    }
    return true;
}

void Decodable::Reset() {
    if (frontend_ != nullptr) frontend_->Reset();
    if (nnet_ != nullptr) nnet_->Reset();
    frame_offset_ = 0;
    frames_ready_ = 0;
    nnet_cache_.Resize(0, 0);
}

}  // namespace ppspeech