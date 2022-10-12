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

#include "decoder/ctc_beam_search_opt.h"
#include "decoder/ctc_prefix_beam_search_score.h"
#include "decoder/decoder_itf.h"

namespace ppspeech {
class ContextGraph;
class CTCPrefixBeamSearch : public DecoderInterface {
  public:
    explicit CTCPrefixBeamSearch(const CTCBeamSearchOptions& opts);
    ~CTCPrefixBeamSearch() {}

    void InitDecoder();

    void Reset();

    void AdvanceDecode(
        const std::shared_ptr<kaldi::DecodableInterface>& decodable);

    std::string GetFinalBestPath();

    std::string GetPartialResult() {
        CHECK(false) << "Not implement.";
        return {};
    }

    void Decode(std::shared_ptr<kaldi::DecodableInterface> decodable);

    std::string GetBestPath();

    std::vector<std::pair<double, std::string>> GetNBestPath();


    int NumFrameDecoded();

    int DecodeLikelihoods(const std::vector<std::vector<BaseFloat>>& probs,
                          std::vector<std::string>& nbest_words);

    const std::vector<float>& ViterbiLikelihood() const {
        return viterbi_likelihood_;
    }

    const std::vector<std::vector<int>>& Inputs() const { return hypotheses_; }

    const std::vector<std::vector<int>>& Outputs() const { return outputs_; }

    const std::vector<float>& Likelihood() const { return likelihood_; }
    const std::vector<std::vector<int>>& Times() const { return times_; }


  private:
    void AdvanceDecoding(const std::vector<std::vector<BaseFloat>>& logp);

    void FinalizeSearch();

    void UpdateOutputs(const std::pair<std::vector<int>, PrefixScore>& prefix);
    void UpdateHypotheses(
        const std::vector<std::pair<std::vector<int>, PrefixScore>>& prefix);
    void UpdateFinalContext();


  private:
    CTCBeamSearchOptions opts_;

    int abs_time_step_ = 0;

    std::unordered_map<std::vector<int>, PrefixScore, PrefixScoreHash>
        cur_hyps_;

    // n-best list and corresponding likelihood, in sorted order
    std::vector<std::vector<int>> hypotheses_;
    std::vector<float> likelihood_;

    std::vector<std::vector<int>> times_;
    std::vector<float> viterbi_likelihood_;

    // Outputs contain the hypotheses_ and tags lik: <context> and </context>
    std::vector<std::vector<int>> outputs_;

    std::shared_ptr<ContextGraph> context_graph_ = nullptr;

    DISALLOW_COPY_AND_ASSIGN(CTCPrefixBeamSearch);
};

}  // namespace ppspeech