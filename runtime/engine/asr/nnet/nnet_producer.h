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

#pragma once

#include "base/common.h"
#include "base/safe_queue.h"
#include "frontend/frontend_itf.h"
#include "nnet/nnet_itf.h"

namespace ppspeech {

class NnetProducer {
  public:
    explicit NnetProducer(std::shared_ptr<NnetBase> nnet,
                          std::shared_ptr<FrontendInterface> frontend,
                          float blank_threshold);
    // Feed feats or waves
    void Accept(const std::vector<kaldi::BaseFloat>& inputs);

    void Acceptlikelihood(const kaldi::Matrix<BaseFloat>& likelihood);

    // nnet
    bool Read(std::vector<kaldi::BaseFloat>* nnet_prob);

    bool Empty() const { return cache_.empty(); }

    void SetInputFinished() {
        LOG(INFO) << "set finished";
        frontend_->SetFinished();
    }

    // the compute thread exit
    bool IsFinished() const { 
        return (frontend_->IsFinished() && finished_); 
    }

    ~NnetProducer() {}

    void Reset() {
        if (frontend_ != NULL) frontend_->Reset();
        if (nnet_ != NULL) nnet_->Reset();
        cache_.clear();
        finished_ = false;
    }

    void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                            float reverse_weight,
                            std::vector<float>* rescoring_score);

    bool Compute();
  private:

    std::shared_ptr<FrontendInterface> frontend_;
    std::shared_ptr<NnetBase> nnet_;
    SafeQueue<std::vector<kaldi::BaseFloat>> cache_;
    std::vector<BaseFloat> last_frame_logprob_;
    bool is_last_frame_skip_ = false;
    int last_max_elem_ = -1;
    float blank_threshold_ = 0.0;
    bool finished_;

    DISALLOW_COPY_AND_ASSIGN(NnetProducer);
};

}  // namespace ppspeech
