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

#include "base/common.h"
#include "frontend/audio/frontend_itf.h"
#include "kaldi/decoder/decodable-itf.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "nnet/nnet_itf.h"

namespace ppspeech {

struct DecodableOpts;

class Decodable : public kaldi::DecodableInterface {
  public:
    explicit Decodable(const std::shared_ptr<NnetInterface>& nnet,
                       const std::shared_ptr<FrontendInterface>& frontend,
                       kaldi::BaseFloat acoustic_scale = 1.0);

    // void Init(DecodableOpts config);

    // nnet logprob output
    virtual kaldi::BaseFloat LogLikelihood(int32 frame, int32 index);

    virtual bool IsLastFrame(int32 frame);

    // nnet output dim, e.g. vocab size
    virtual int32 NumIndices() const;

    // nnet prob output
    virtual bool FrameLikelihood(int32 frame,
                                 std::vector<kaldi::BaseFloat>* likelihood);

    virtual int32 NumFramesReady() const;

    // for offline test
    void Acceptlikelihood(const kaldi::Matrix<kaldi::BaseFloat>& likelihood);

    void Reset();

    bool IsInputFinished() const { return frontend_->IsFinished(); }

    bool EnsureFrameHaveComputed(int32 frame);

    int32 TokenId2NnetId(int32 token_id);

  private:
    bool AdvanceChunk();

    std::shared_ptr<FrontendInterface> frontend_;
    std::shared_ptr<NnetInterface> nnet_;

    // nnet outputs' cache
    kaldi::Matrix<kaldi::BaseFloat> nnet_out_cache_;

    // the frame is nnet prob frame rather than audio feature frame
    // nnet frame subsample the feature frame
    // eg: 35 frame features output 8 frame inferences
    int32 frame_offset_;
    int32 frames_ready_;

    // todo: feature frame mismatch with nnet inference frame
    // so use subsampled_frame
    int32 current_log_post_subsampled_offset_;
    int32 num_chunk_computed_;

    kaldi::BaseFloat acoustic_scale_;
};

}  // namespace ppspeech
