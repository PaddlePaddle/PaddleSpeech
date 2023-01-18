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
#include "matrix/kaldi-matrix.h"

namespace ppspeech {

using std::vector;
using kaldi::BaseFloat;

NnetProducer::NnetProducer(std::shared_ptr<NnetBase> nnet,
                           std::shared_ptr<FrontendInterface> frontend)
    : nnet_(nnet), frontend_(frontend) {
        abort_ = false;
        Reset();
        thread_ = std::thread(RunNnetEvaluation, this);
    }

void NnetProducer::Accept(const std::vector<kaldi::BaseFloat>& inputs) {
    frontend_->Accept(inputs);
    condition_variable_.notify_one();
}

void NnetProducer::UnLock() {
    std::unique_lock<std::mutex> lock(read_mutex_);
    while (frontend_->IsFinished() == false && cache_.empty()) {
        condition_read_ready_.wait(lock); 
    }
    return;
}

void NnetProducer::RunNnetEvaluation(NnetProducer *me) {
    me->RunNnetEvaluationInteral();
}

void NnetProducer::RunNnetEvaluationInteral() {
    bool result = false;
    LOG(INFO) << "NnetEvaluationInteral begin";
    while (!abort_) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_variable_.wait(lock);
        do {
            result = Compute();
        } while (result);
        if (frontend_->IsFinished() == true) {
           if (cache_.empty()) finished_ = true;
        }
    }
    LOG(INFO) << "NnetEvaluationInteral exit";
}

void NnetProducer::Acceptlikelihood(
    const kaldi::Matrix<BaseFloat>& likelihood) {
    std::vector<BaseFloat> prob;
    prob.resize(likelihood.NumCols());
    for (size_t idx = 0; idx < likelihood.NumRows(); ++idx) {
        for (size_t col = 0; col < likelihood.NumCols(); ++col) {
            prob[col] = likelihood(idx, col);
        }
        cache_.push_back(prob);
    }
}

bool NnetProducer::Read(std::vector<kaldi::BaseFloat>* nnet_prob) {
    bool flag = cache_.pop(nnet_prob);
    condition_variable_.notify_one();
    return flag;
}

bool NnetProducer::ReadandCompute(std::vector<kaldi::BaseFloat>* nnet_prob) {
    Compute();
    if (frontend_->IsFinished() && cache_.empty()) finished_ = true;
    bool flag = cache_.pop(nnet_prob);
    return flag;
}

bool NnetProducer::Compute() {
    vector<BaseFloat> features;
    if (frontend_ == NULL || frontend_->Read(&features) == false) {
        // no feat or frontend_ not init.
        VLOG(2) << "no feat avalible";
        return false;
    }
    CHECK_GE(frontend_->Dim(), 0);
    VLOG(1) << "Forward in " << features.size() / frontend_->Dim() << " feats.";

    NnetOut out;
    nnet_->FeedForward(features, frontend_->Dim(), &out);
    int32& vocab_dim = out.vocab_dim;
    size_t nframes = out.logprobs.size() / vocab_dim;
    VLOG(1) << "Forward out " << nframes << " decoder frames.";
    for (size_t idx = 0; idx < nframes; ++idx) {
        std::vector<BaseFloat> logprob(
            out.logprobs.data() + idx * vocab_dim,
            out.logprobs.data() + (idx + 1) * vocab_dim);
        cache_.push_back(logprob);
        condition_read_ready_.notify_one();
    }
    return true;
}

void NnetProducer::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                      float reverse_weight,
                                      std::vector<float>* rescoring_score) {
    nnet_->AttentionRescoring(hyps, reverse_weight, rescoring_score);
}

}  // namespace ppspeech
