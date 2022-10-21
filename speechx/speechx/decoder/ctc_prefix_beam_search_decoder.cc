// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Binbin Zhang (binbzha@qq.com)
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


#include "decoder/ctc_prefix_beam_search_decoder.h"

#include "absl/strings/str_join.h"
#include "base/common.h"
#include "decoder/ctc_beam_search_opt.h"
#include "decoder/ctc_prefix_beam_search_score.h"
#include "utils/math.h"

#ifdef USE_PROFILING
#include "paddle/fluid/platform/profiler.h"
using paddle::platform::RecordEvent;
using paddle::platform::TracerEventType;
#endif

namespace ppspeech {

CTCPrefixBeamSearch::CTCPrefixBeamSearch(const std::string vocab_path,
                                         const CTCBeamSearchOptions& opts)
    : opts_(opts) {
    unit_table_ = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(vocab_path));
    CHECK(unit_table_ != nullptr);

    Reset();
}

void CTCPrefixBeamSearch::Reset() {
    num_frame_decoded_ = 0;

    cur_hyps_.clear();

    hypotheses_.clear();
    likelihood_.clear();
    viterbi_likelihood_.clear();
    times_.clear();
    outputs_.clear();

    // empty hyp with Score
    std::vector<int> empty;
    PrefixScore prefix_score;
    prefix_score.b = 0.0f;             // log(1)
    prefix_score.nb = -kBaseFloatMax;  // log(0)
    prefix_score.v_b = 0.0f;           // log(1)
    prefix_score.v_nb = 0.0f;          // log(1)
    cur_hyps_[empty] = prefix_score;

    outputs_.emplace_back(empty);
    hypotheses_.emplace_back(empty);
    likelihood_.emplace_back(prefix_score.TotalScore());
    times_.emplace_back(empty);
}

void CTCPrefixBeamSearch::InitDecoder() { Reset(); }


void CTCPrefixBeamSearch::AdvanceDecode(
    const std::shared_ptr<kaldi::DecodableInterface>& decodable) {
    while (1) {
        // forward frame by frame
        std::vector<kaldi::BaseFloat> frame_prob;
        bool flag = decodable->FrameLikelihood(num_frame_decoded_, &frame_prob);
        if (flag == false) {
            LOG(INFO) << "decoder advance decode exit." << frame_prob.size();
            break;
        }

        std::vector<std::vector<kaldi::BaseFloat>> likelihood;
        likelihood.push_back(frame_prob);
        AdvanceDecoding(likelihood);
        VLOG(2) << "num_frame_decoded_: " << num_frame_decoded_;
    }
}

static bool PrefixScoreCompare(
    const std::pair<std::vector<int>, PrefixScore>& a,
    const std::pair<std::vector<int>, PrefixScore>& b) {
    // log domain
    return a.second.TotalScore() > b.second.TotalScore();
}


void CTCPrefixBeamSearch::AdvanceDecoding(
    const std::vector<std::vector<kaldi::BaseFloat>>& logp) {
#ifdef USE_PROFILING
    RecordEvent event("CtcPrefixBeamSearch::AdvanceDecoding",
                      TracerEventType::UserDefined,
                      1);
#endif

    if (logp.size() == 0) return;

    int first_beam_size =
        std::min(static_cast<int>(logp[0].size()), opts_.first_beam_size);

    for (int t = 0; t < logp.size(); ++t, ++num_frame_decoded_) {
        const std::vector<float>& logp_t = logp[t];
        std::unordered_map<std::vector<int>, PrefixScore, PrefixScoreHash>
            next_hyps;

        // 1. first beam prune, only select topk candidates
        std::vector<float> topk_score;
        std::vector<int32_t> topk_index;
        TopK(logp_t, first_beam_size, &topk_score, &topk_index);
        VLOG(2) << "topk: " << num_frame_decoded_ << " "
                << *std::max_element(logp_t.begin(), logp_t.end()) << " "
                << topk_score[0];
        for (int i = 0; i < topk_score.size(); i++) {
            VLOG(2) << "topk: " << num_frame_decoded_ << " " << topk_score[i];
        }

        // 2. token passing
        for (int i = 0; i < topk_index.size(); ++i) {
            int id = topk_index[i];
            auto prob = topk_score[i];

            for (const auto& it : cur_hyps_) {
                const std::vector<int>& prefix = it.first;
                const PrefixScore& prefix_score = it.second;

                // If prefix doesn't exist in next_hyps, next_hyps[prefix] will
                // insert
                // PrefixScore(-inf, -inf) by default, since the default
                // constructor
                // of PrefixScore will set fields b(blank ending Score) and
                // nb(none blank ending Score) to -inf, respectively.

                if (id == opts_.blank) {
                    // case 0: *a + <blank> => *a, *a<blank> + <blank> => *a,
                    // prefix not
                    // change
                    PrefixScore& next_score = next_hyps[prefix];
                    next_score.b =
                        LogSumExp(next_score.b, prefix_score.Score() + prob);

                    // timestamp, blank is slince, not effact timestamp
                    next_score.v_b = prefix_score.ViterbiScore() + prob;
                    next_score.times_b = prefix_score.Times();

                    // Prefix not changed, copy the context from pefix
                    if (context_graph_ && !next_score.has_context) {
                        next_score.CopyContext(prefix_score);
                        next_score.has_context = true;
                    }

                } else if (!prefix.empty() && id == prefix.back()) {
                    // case 1: *a + a => *a, prefix not changed
                    PrefixScore& next_score1 = next_hyps[prefix];
                    next_score1.nb =
                        LogSumExp(next_score1.nb, prefix_score.nb + prob);

                    // timestamp, non-blank symbol effact timestamp
                    if (next_score1.v_nb < prefix_score.v_nb + prob) {
                        // compute viterbi Score
                        next_score1.v_nb = prefix_score.v_nb + prob;
                        if (next_score1.cur_token_prob < prob) {
                            // store max token prob
                            next_score1.cur_token_prob = prob;
                            // update this timestamp as token appeared here.
                            next_score1.times_nb = prefix_score.times_nb;
                            assert(next_score1.times_nb.size() > 0);
                            next_score1.times_nb.back() = num_frame_decoded_;
                        }
                    }

                    // Prefix not changed, copy the context from pefix
                    if (context_graph_ && !next_score1.has_context) {
                        next_score1.CopyContext(prefix_score);
                        next_score1.has_context = true;
                    }

                    // case 2: *a<blank> + a => *aa, prefix changed.
                    std::vector<int> new_prefix(prefix);
                    new_prefix.emplace_back(id);
                    PrefixScore& next_score2 = next_hyps[new_prefix];
                    next_score2.nb =
                        LogSumExp(next_score2.nb, prefix_score.b + prob);

                    // timestamp, non-blank symbol effact timestamp
                    if (next_score2.v_nb < prefix_score.v_b + prob) {
                        // compute viterbi Score
                        next_score2.v_nb = prefix_score.v_b + prob;
                        // new token added
                        next_score2.cur_token_prob = prob;
                        next_score2.times_nb = prefix_score.times_b;
                        next_score2.times_nb.emplace_back(num_frame_decoded_);
                    }

                    // Prefix changed, calculate the context Score.
                    if (context_graph_ && !next_score2.has_context) {
                        next_score2.UpdateContext(
                            context_graph_, prefix_score, id, prefix.size());
                        next_score2.has_context = true;
                    }

                } else {
                    // id != prefix.back()
                    // case 3: *a + b => *ab, *a<blank> +b => *ab
                    std::vector<int> new_prefix(prefix);
                    new_prefix.emplace_back(id);
                    PrefixScore& next_score = next_hyps[new_prefix];
                    next_score.nb =
                        LogSumExp(next_score.nb, prefix_score.Score() + prob);

                    // timetamp, non-blank symbol effact timestamp
                    if (next_score.v_nb < prefix_score.ViterbiScore() + prob) {
                        next_score.v_nb = prefix_score.ViterbiScore() + prob;

                        next_score.cur_token_prob = prob;
                        next_score.times_nb = prefix_score.Times();
                        next_score.times_nb.emplace_back(num_frame_decoded_);
                    }

                    // Prefix changed, calculate the context Score.
                    if (context_graph_ && !next_score.has_context) {
                        next_score.UpdateContext(
                            context_graph_, prefix_score, id, prefix.size());
                        next_score.has_context = true;
                    }
                }
            }  // end for (const auto& it : cur_hyps_)
        }      // end for (int i = 0; i < topk_index.size(); ++i)

        // 3. second beam prune, only keep top n best paths
        std::vector<std::pair<std::vector<int>, PrefixScore>> arr(
            next_hyps.begin(), next_hyps.end());
        int second_beam_size =
            std::min(static_cast<int>(arr.size()), opts_.second_beam_size);
        std::nth_element(arr.begin(),
                         arr.begin() + second_beam_size,
                         arr.end(),
                         PrefixScoreCompare);
        arr.resize(second_beam_size);
        std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

        // 4. update cur_hyps by next_hyps, and get new result
        UpdateHypotheses(arr);
    }  // end for (int t = 0; t < logp.size(); ++t, ++num_frame_decoded_)
}


void CTCPrefixBeamSearch::UpdateHypotheses(
    const std::vector<std::pair<std::vector<int>, PrefixScore>>& hyps) {
    cur_hyps_.clear();

    outputs_.clear();
    hypotheses_.clear();
    likelihood_.clear();
    viterbi_likelihood_.clear();
    times_.clear();

    for (auto& item : hyps) {
        cur_hyps_[item.first] = item.second;

        UpdateOutputs(item);
        hypotheses_.emplace_back(std::move(item.first));
        likelihood_.emplace_back(item.second.TotalScore());
        viterbi_likelihood_.emplace_back(item.second.ViterbiScore());
        times_.emplace_back(item.second.Times());
    }
}

void CTCPrefixBeamSearch::UpdateOutputs(
    const std::pair<std::vector<int>, PrefixScore>& prefix) {
    const std::vector<int>& input = prefix.first;
    const std::vector<int>& start_boundaries = prefix.second.start_boundaries;
    const std::vector<int>& end_boundaries = prefix.second.end_boundaries;

    // add <context> </context> tag
    std::vector<int> output;
    int s = 0;
    int e = 0;
    for (int i = 0; i < input.size(); ++i) {
        // if (s < start_boundaries.size() && i == start_boundaries[s]){
        //     // <context>
        //     output.emplace_back(context_graph_->start_tag_id());
        //     ++s;
        // }

        output.emplace_back(input[i]);

        // if (e < end_boundaries.size() && i == end_boundaries[e]){
        //     // </context>
        //     output.emplace_back(context_graph_->end_tag_id());
        //     ++e;
        // }
    }

    outputs_.emplace_back(output);
}

void CTCPrefixBeamSearch::FinalizeSearch() {
    UpdateFinalContext();

    VLOG(2) << "num_frame_decoded_: " << num_frame_decoded_;
    int cnt = 0;
    for (int i = 0; i < hypotheses_.size(); i++) {
        VLOG(2) << "hyp " << cnt << " len: " << hypotheses_[i].size()
                << " ctc score: " << likelihood_[i];
        for (int j = 0; j < hypotheses_[i].size(); j++) {
            VLOG(2) << hypotheses_[i][j];
        }
    }
}

void CTCPrefixBeamSearch::UpdateFinalContext() {
    if (context_graph_ == nullptr) return;

    CHECK(hypotheses_.size() == cur_hyps_.size());
    CHECK(hypotheses_.size() == likelihood_.size());

    // We should backoff the context Score/state when the context is
    // not fully matched at the last time.
    for (const auto& prefix : hypotheses_) {
        PrefixScore& prefix_score = cur_hyps_[prefix];
        if (prefix_score.context_score != 0) {
            prefix_score.UpdateContext(
                context_graph_, prefix_score, 0, prefix.size());
        }
    }
    std::vector<std::pair<std::vector<int>, PrefixScore>> arr(cur_hyps_.begin(),
                                                              cur_hyps_.end());
    std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

    // Update cur_hyps_ and get new result
    UpdateHypotheses(arr);
}

std::string CTCPrefixBeamSearch::GetBestPath(int index) {
    int n_hyps = Outputs().size();
    CHECK(n_hyps > 0);
    CHECK(index < n_hyps);
    std::vector<int> one = Outputs()[index];
    std::string sentence;
    for (int i = 0; i < one.size(); i++) {
        sentence += unit_table_->Find(one[i]);
    }
    return sentence;
}

std::string CTCPrefixBeamSearch::GetBestPath() { return GetBestPath(0); }

std::vector<std::pair<double, std::string>> CTCPrefixBeamSearch::GetNBestPath(
    int n) {
    int hyps_size = hypotheses_.size();
    CHECK(hyps_size > 0);

    int min_n = n == -1 ? hypotheses_.size() : std::min(n, hyps_size);

    std::vector<std::pair<double, std::string>> n_best;
    n_best.reserve(min_n);

    for (int i = 0; i < min_n; i++) {
        n_best.emplace_back(Likelihood()[i], GetBestPath(i));
    }
    return n_best;
}

std::vector<std::pair<double, std::string>>
CTCPrefixBeamSearch::GetNBestPath() {
    return GetNBestPath(-1);
}

std::string CTCPrefixBeamSearch::GetFinalBestPath() { return GetBestPath(); }

std::string CTCPrefixBeamSearch::GetPartialResult() { return GetBestPath(); }


}  // namespace ppspeech