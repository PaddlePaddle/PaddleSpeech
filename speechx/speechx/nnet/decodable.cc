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
                     const std::shared_ptr<FrontendInterface>& frontend,
                     kaldi::BaseFloat acoustic_scale)
    : frontend_(frontend),
      nnet_(nnet),
      frame_offset_(0),
      frames_ready_(0),
      acoustic_scale_(acoustic_scale) {}

// for debug
void Decodable::Acceptlikelihood(const Matrix<BaseFloat>& likelihood) {
    nnet_out_cache_ = likelihood;
    frames_ready_ += likelihood.NumRows();
}

// Decodable::Init(DecodableConfig config) {
//}

// return the size of frame have computed.
int32 Decodable::NumFramesReady() const { return frames_ready_; }


// frame idx is from 0 to frame_ready_ -1;
bool Decodable::IsLastFrame(int32 frame) {
    bool flag = EnsureFrameHaveComputed(frame);
    return frame >= frames_ready_;
}

int32 Decodable::NumIndices() const { return 0; }

// the ilable(TokenId) of wfst(TLG) insert <eps>(id = 0) in front of Nnet prob
// id.
int32 Decodable::TokenId2NnetId(int32 token_id) { return token_id - 1; }

BaseFloat Decodable::LogLikelihood(int32 frame, int32 index) {
    CHECK_LE(index, nnet_out_cache_.NumCols());
    CHECK_LE(frame, frames_ready_);
    int32 frame_idx = frame - frame_offset_;
    // the nnet output is prob ranther than log prob
    // the index - 1, because the ilabel
    return acoustic_scale_ *
           std::log(nnet_out_cache_(frame_idx, TokenId2NnetId(index)) +
                    std::numeric_limits<float>::min());
}

bool Decodable::EnsureFrameHaveComputed(int32 frame) {
    if (frame >= frames_ready_) {
        return AdvanceChunk();
    }
    return true;
}

bool Decodable::AdvanceChunk() {
    // read feats
    Vector<BaseFloat> features;
    if (frontend_ == NULL || frontend_->Read(&features) == false) {
        // no feat or frontend_ not init.
        return false;
    }

    // forward feats
    NnetOut out;
    nnet_->FeedForward(features, frontend_->Dim(), &out);
    int32& vocab_dim = out.vocab_dim;
    Vector<BaseFloat>& probs = out.logprobs;

    // cache nnet outupts
    nnet_out_cache_.Resize(probs.Dim() / vocab_dim, vocab_dim);
    nnet_out_cache_.CopyRowsFromVec(probs);

    // update state
    frame_offset_ = frames_ready_;
    frames_ready_ += nnet_out_cache_.NumRows();
    return true;
}

// read one frame likelihood
bool Decodable::FrameLikelihood(int32 frame, vector<BaseFloat>* likelihood) {
    if (EnsureFrameHaveComputed(frame) == false) {
        return false;
    }

    int vocab_size = nnet_out_cache_.NumCols();
    likelihood->resize(vocab_size);

    for (int32 idx = 0; idx < vocab_size; ++idx) {
        (*likelihood)[idx] =
            nnet_out_cache_(frame - frame_offset_, idx) * acoustic_scale_;
    }
    return true;
}

void Decodable::Reset() {
    if (frontend_ != nullptr) frontend_->Reset();
    if (nnet_ != nullptr) nnet_->Reset();
    frame_offset_ = 0;
    frames_ready_ = 0;
    nnet_out_cache_.Resize(0, 0);
}

}  // namespace ppspeech