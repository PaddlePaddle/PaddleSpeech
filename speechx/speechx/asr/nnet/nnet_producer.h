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
                          std::shared_ptr<FrontendInterface> frontend = NULL);

    // Feed feats or waves
    void Accept(const std::vector<kaldi::BaseFloat>& inputs);

    void Acceptlikelihood(const kaldi::Matrix<BaseFloat>& likelihood);

    // nnet
    bool Read(std::vector<kaldi::BaseFloat>* nnet_prob);

    bool Empty() const { return cache_.empty(); }

    void SetFinished() {
        LOG(INFO) << "set finished";
        // std::unique_lock<std::mutex> lock(mutex_);
        frontend_->SetFinished();

        // read the last chunk data
        Compute();
        // ready_feed_condition_.notify_one();
        LOG(INFO) << "compute last feats done.";
    }

    bool IsFinished() const { return frontend_->IsFinished(); }

    void Reset() {
        frontend_->Reset();
        nnet_->Reset();
        VLOG(3) << "feature cache reset: cache size: " << cache_.size();
        cache_.clear();
    }

    void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                            float reverse_weight,
                            std::vector<float>* rescoring_score);

  private:
    bool Compute();

    std::shared_ptr<FrontendInterface> frontend_;
    std::shared_ptr<NnetBase> nnet_;
    SafeQueue<std::vector<kaldi::BaseFloat>> cache_;

    DISALLOW_COPY_AND_ASSIGN(NnetProducer);
};

}  // namespace ppspeech
