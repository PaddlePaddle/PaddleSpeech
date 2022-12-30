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

#include "nnet/nnet_producer.h"

namespace ppspeech {

using kaldi::Vector;
using kaldi::BaseFloat;

NnetProducer::NnetProducer(std::shared_ptr<NnetBase> nnet,
                           std::shared_ptr<FrontendInterface> frontend)
    : nnet_(nnet), frontend_(frontend) {}

void NnetProducer::Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs) {
    frontend_->Accept(inputs);
    bool result = false;
    do {
        result = Compute();
    } while (result);
}

void NnetProducer::Acceptlikelihood(
    const kaldi::Matrix<BaseFloat>& likelihood) {
    std::vector<BaseFloat> prob;
    prob.resize(likelihood.NumCols());
    for (size_t idx = 0; idx < likelihood.NumRows(); ++idx) {
        for (size_t col = 0; col < likelihood.NumCols(); ++col) {
            prob[col] = likelihood(idx, col);
            cache_.push_back(prob);
        }
    }
}

bool NnetProducer::Read(std::vector<kaldi::BaseFloat>* nnet_prob) {
    bool flag = cache_.pop(nnet_prob);
    return flag;
}

bool NnetProducer::Compute() {
    Vector<BaseFloat> features;
    if (frontend_ == NULL || frontend_->Read(&features) == false) {
        // no feat or frontend_ not init.
        VLOG(3) << "no feat avalible";
        return false;
    }
    CHECK_GE(frontend_->Dim(), 0);
    VLOG(2) << "Forward in " << features.Dim() / frontend_->Dim() << " feats.";

    NnetOut out;
    nnet_->FeedForward(features, frontend_->Dim(), &out);
    int32& vocab_dim = out.vocab_dim;
    Vector<BaseFloat>& logprobs = out.logprobs;
    size_t nframes = logprobs.Dim() / vocab_dim;
    VLOG(2) << "Forward out " << nframes << " decoder frames.";
    std::vector<BaseFloat> logprob(vocab_dim);
    for (size_t idx = 0; idx < nframes; ++idx) {
        for (size_t prob_idx = 0; prob_idx < vocab_dim; ++prob_idx) {
            logprob[prob_idx] = logprobs(idx * vocab_dim + prob_idx);
        }
        cache_.push_back(logprob);
    }
    return true;
}

void NnetProducer::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                      float reverse_weight,
                                      std::vector<float>* rescoring_score) {
    nnet_->AttentionRescoring(hyps, reverse_weight, rescoring_score);
}

}  // namespace ppspeech