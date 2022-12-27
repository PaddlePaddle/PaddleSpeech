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
using kaldi::Vector;
using std::vector;

Decodable::Decodable(const std::shared_ptr<NnetProducer>& nnet_producer,
                     kaldi::BaseFloat acoustic_scale)
    : nnet_producer_(nnet_producer),
      frame_offset_(0),
      frames_ready_(0),
      acoustic_scale_(acoustic_scale) {}

// for debug
void Decodable::Acceptlikelihood(const Matrix<BaseFloat>& likelihood) {
    nnet_producer_->Acceptlikelihood(likelihood);
}


// return the size of frame have computed.
int32 Decodable::NumFramesReady() const { return frames_ready_; }


// frame idx is from 0 to frame_ready_ -1;
bool Decodable::IsLastFrame(int32 frame) {
    EnsureFrameHaveComputed(frame);
    return frame >= frames_ready_;
}

int32 Decodable::NumIndices() const { return 0; }

// the ilable(TokenId) of wfst(TLG) insert <eps>(id = 0) in front of Nnet prob
// id.
int32 Decodable::TokenId2NnetId(int32 token_id) { return token_id - 1; }


bool Decodable::EnsureFrameHaveComputed(int32 frame) {
    // decoding frame
    if (frame >= frames_ready_) {
        return AdvanceChunk();
    }
    return true;
}

bool Decodable::AdvanceChunk() {
    kaldi::Timer timer;
    bool flag = nnet_producer_->Read(&framelikelihood_);
    if (flag == false) return false;
    frame_offset_ = frames_ready_;
    frames_ready_ += 1;
    VLOG(1) << "AdvanceChunk feat + forward cost: " << timer.Elapsed()
            << " sec.";
    return true;
}

bool Decodable::AdvanceChunk(kaldi::Vector<kaldi::BaseFloat>* logprobs,
                             int* vocab_dim) {
    if (AdvanceChunk() == false) {
        return false;
    }

    if (framelikelihood_.empty()) {
        LOG(WARNING) << "No new nnet out in cache.";
        return false;
    }

    size_t dim = framelikelihood_.size();
    logprobs->Resize(framelikelihood_.size());
    std::memcpy(logprobs->Data(),
                framelikelihood_.data(),
                dim * sizeof(kaldi::BaseFloat));
    *vocab_dim = framelikelihood_.size();
    return true;
}

// read one frame likelihood
bool Decodable::FrameLikelihood(int32 frame, vector<BaseFloat>* likelihood) {
    if (EnsureFrameHaveComputed(frame) == false) {
        VLOG(3) << "framelikehood exit.";
        return false;
    }

    CHECK_EQ(1, (frames_ready_ - frame_offset_));
    *likelihood = framelikelihood_;
    return true;
}

BaseFloat Decodable::LogLikelihood(int32 frame, int32 index) {
    if (EnsureFrameHaveComputed(frame) == false) {
        return false;
    }

    CHECK_LE(index, framelikelihood_.size());
    CHECK_LE(frame, frames_ready_);

    // the nnet output is prob ranther than log prob
    // the index - 1, because the ilabel
    BaseFloat logprob = 0.0;
    int32 frame_idx = frame - frame_offset_;
    CHECK_EQ(frame_idx, 0);
    logprob = framelikelihood_[TokenId2NnetId(index)];
    return acoustic_scale_ * logprob;
}

void Decodable::Reset() {
    if (nnet_producer_ != nullptr) nnet_producer_->Reset();
    frame_offset_ = 0;
    frames_ready_ = 0;
    framelikelihood_.clear();
}

void Decodable::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                   float reverse_weight,
                                   std::vector<float>* rescoring_score) {
    kaldi::Timer timer;
    nnet_producer_->AttentionRescoring(hyps, reverse_weight, rescoring_score);
    VLOG(1) << "Attention Rescoring cost:  " << timer.Elapsed() << " sec.";
}

}  // namespace ppspeech
