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
#include "frontend/feature_extractor_interface.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "nnet/decodable-itf.h"
#include "nnet/nnet_interface.h"

namespace ppspeech {

struct DecodableOpts;

class Decodable : public kaldi::DecodableInterface {
  public:
    explicit Decodable(const std::shared_ptr<NnetInterface>& nnet);
    // void Init(DecodableOpts config);
    virtual kaldi::BaseFloat LogLikelihood(int32 frame, int32 index);
    virtual bool IsLastFrame(int32 frame) const;
    virtual int32 NumIndices() const;
    virtual std::vector<BaseFloat> FrameLogLikelihood(int32 frame);
    void Acceptlikelihood(
        const kaldi::Matrix<kaldi::BaseFloat>& likelihood);  // remove later
    void FeedFeatures(const kaldi::Matrix<kaldi::BaseFloat>&
                          feature);  // only for test, todo remove later
    void Reset();
    void InputFinished() { finished_ = true; }

  private:
    std::shared_ptr<FeatureExtractorInterface> frontend_;
    std::shared_ptr<NnetInterface> nnet_;
    kaldi::Matrix<kaldi::BaseFloat> nnet_cache_;
    bool finished_;
    int32 frames_ready_;
};

}  // namespace ppspeech
