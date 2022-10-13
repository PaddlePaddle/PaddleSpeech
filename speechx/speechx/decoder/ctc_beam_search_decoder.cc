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
#include "decoder/ctc_decoders/decoder_utils.h"
#include "decoder/ctc_beam_search_decoder.h"
#include "utils/file_utils.h"

namespace ppspeech {

using std::vector;
using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

CTCBeamSearch::CTCBeamSearch(const CTCBeamSearchOptions& opts)
    : opts_(opts),
      init_ext_scorer_(nullptr),
      space_id_(-1),
      root_(nullptr) {
    LOG(INFO) << "dict path: " << opts_.dict_file;
    if (!ReadFileToVector(opts_.dict_file, &vocabulary_)) {
        LOG(INFO) << "load the dict failed";
    }
    LOG(INFO) << "read the vocabulary success, dict size: "
              << vocabulary_.size();

    LOG(INFO) << "language model path: " << opts_.lm_path;
    if (opts_.lm_path != "") {
        init_ext_scorer_ = std::make_shared<Scorer>(
            opts_.alpha, opts_.beta, opts_.lm_path, vocabulary_);
    }

    CHECK(opts_.blank==0);

    auto it = std::find(vocabulary_.begin(), vocabulary_.end(), " ");
    space_id_ = it - vocabulary_.begin();
    // if no space in vocabulary
    if ((size_t)space_id_ >= vocabulary_.size()) {
        space_id_ = -2;
    }
}

void CTCBeamSearch::Reset() {
    // num_frame_decoded_ = 0;
    // ResetPrefixes();
    InitDecoder();
}

void CTCBeamSearch::InitDecoder() {
    num_frame_decoded_ = 0;
    // ResetPrefixes();
    prefixes_.clear();

    root_ = std::make_shared<PathTrie>();
    root_->score = root_->log_prob_b_prev = 0.0;
    prefixes_.push_back(root_.get());
    if (init_ext_scorer_ != nullptr &&
        !init_ext_scorer_->is_character_based()) {
        auto fst_dict =
            static_cast<fst::StdVectorFst*>(init_ext_scorer_->dictionary);
        fst::StdVectorFst* dict_ptr = fst_dict->Copy(true);
        root_->set_dictionary(dict_ptr);

        auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
        root_->set_matcher(matcher);
    }
}

void CTCBeamSearch::Decode(
    std::shared_ptr<kaldi::DecodableInterface> decodable) {
    return;
}

// todo rename, refactor
void CTCBeamSearch::AdvanceDecode(
    const std::shared_ptr<kaldi::DecodableInterface>& decodable) {
    while (1) {
        vector<vector<BaseFloat>> likelihood;
        vector<BaseFloat> frame_prob;
        bool flag = decodable->FrameLikelihood(num_frame_decoded_, &frame_prob);
        if (flag == false) break;
        likelihood.push_back(frame_prob);
        AdvanceDecoding(likelihood);
    }
}

void CTCBeamSearch::ResetPrefixes() {
    for (size_t i = 0; i < prefixes_.size(); i++) {
        if (prefixes_[i] != nullptr) {
            delete prefixes_[i];
            prefixes_[i] = nullptr;
        }
    }
    prefixes_.clear();
}

int CTCBeamSearch::DecodeLikelihoods(const vector<vector<float>>& probs,
                                     vector<string>& nbest_words) {
    kaldi::Timer timer;
    AdvanceDecoding(probs);
    LOG(INFO) << "ctc decoding elapsed time(s) "
              << static_cast<float>(timer.Elapsed()) / 1000.0f;
    return 0;
}

vector<std::pair<double, string>> CTCBeamSearch::GetNBestPath(int n) {
    int beam_size = n == -1 ?  opts_.beam_size: std::min(n, opts_.beam_size);
    return get_beam_search_result(prefixes_, vocabulary_, beam_size);
}

vector<std::pair<double, string>> CTCBeamSearch::GetNBestPath() {
    return GetNBestPath(-1);
}

string CTCBeamSearch::GetBestPath() {
    std::vector<std::pair<double, std::string>> result;
    result = get_beam_search_result(prefixes_, vocabulary_, opts_.beam_size);
    return result[0].second;
}

string CTCBeamSearch::GetFinalBestPath() {
    CalculateApproxScore();
    LMRescore();
    return GetBestPath();
}

void CTCBeamSearch::AdvanceDecoding(const vector<vector<BaseFloat>>& probs) {
    size_t num_time_steps = probs.size();
    size_t beam_size = opts_.beam_size;
    double cutoff_prob = opts_.cutoff_prob;
    size_t cutoff_top_n = opts_.cutoff_top_n;

    vector<vector<double>> probs_seq(probs.size(),
                                     vector<double>(probs[0].size(), 0));

    int row = probs.size();
    int col = probs[0].size();
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            probs_seq[i][j] = static_cast<double>(probs[i][j]);
        }
    }

    for (size_t time_step = 0; time_step < num_time_steps; time_step++) {
        const auto& prob = probs_seq[time_step];

        float min_cutoff = -NUM_FLT_INF;
        bool full_beam = false;
        if (init_ext_scorer_ != nullptr) {
            size_t num_prefixes_ = std::min(prefixes_.size(), beam_size);
            std::sort(prefixes_.begin(),
                      prefixes_.begin() + num_prefixes_,
                      prefix_compare);

            if (num_prefixes_ == 0) {
                continue;
            }
            min_cutoff = prefixes_[num_prefixes_ - 1]->score +
                         std::log(prob[opts_.blank]) -
                         std::max(0.0, init_ext_scorer_->beta);

            full_beam = (num_prefixes_ == beam_size);
        }

        vector<std::pair<size_t, float>> log_prob_idx =
            get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n);

        // loop over chars
        size_t log_prob_idx_len = log_prob_idx.size();
        for (size_t index = 0; index < log_prob_idx_len; index++) {
            SearchOneChar(full_beam, log_prob_idx[index], min_cutoff);
        }

        prefixes_.clear();

        // update log probs
        root_->iterate_to_vec(prefixes_);
        // only preserve top beam_size prefixes_
        if (prefixes_.size() >= beam_size) {
            std::nth_element(prefixes_.begin(),
                             prefixes_.begin() + beam_size,
                             prefixes_.end(),
                             prefix_compare);
            for (size_t i = beam_size; i < prefixes_.size(); ++i) {
                prefixes_[i]->remove();
            }
        }  // end if
        num_frame_decoded_++;
    }  // end for probs_seq
}

int32 CTCBeamSearch::SearchOneChar(
    const bool& full_beam,
    const std::pair<size_t, BaseFloat>& log_prob_idx,
    const BaseFloat& min_cutoff) {
    size_t beam_size = opts_.beam_size;
    const auto& c = log_prob_idx.first;
    const auto& log_prob_c = log_prob_idx.second;
    size_t prefixes_len = std::min(prefixes_.size(), beam_size);

    for (size_t i = 0; i < prefixes_len; ++i) {
        auto prefix = prefixes_[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
            break;
        }

        if (c == opts_.blank) {
            prefix->log_prob_b_cur =
                log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
            continue;
        }

        // repeated character
        if (c == prefix->character) {
            // p_{nb}(l;x_{1:t}) = p(c;x_{t})p(l;x_{1:t-1})
            prefix->log_prob_nb_cur = log_sum_exp(
                prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }

        // get new prefix
        auto prefix_new = prefix->get_path_trie(c);
        if (prefix_new != nullptr) {
            float log_p = -NUM_FLT_INF;
            if (c == prefix->character &&
                prefix->log_prob_b_prev > -NUM_FLT_INF) {
                // p_{nb}(l^{+};x_{1:t}) = p(c;x_{t})p_{b}(l;x_{1:t-1})
                log_p = log_prob_c + prefix->log_prob_b_prev;
            } else if (c != prefix->character) {
                // p_{nb}(l^{+};x_{1:t}) = p(c;x_{t}) p(l;x_{1:t-1})
                log_p = log_prob_c + prefix->score;
            }

            // language model scoring
            if (init_ext_scorer_ != nullptr &&
                (c == space_id_ || init_ext_scorer_->is_character_based())) {
                PathTrie* prefix_to_score = nullptr;
                // skip scoring the space
                if (init_ext_scorer_->is_character_based()) {
                    prefix_to_score = prefix_new;
                } else {
                    prefix_to_score = prefix;
                }

                float score = 0.0;
                vector<string> ngram;
                ngram = init_ext_scorer_->make_ngram(prefix_to_score);
                // lm score: p_{lm}(W)^{\alpha} + \beta
                score = init_ext_scorer_->get_log_cond_prob(ngram) *
                        init_ext_scorer_->alpha;
                log_p += score;
                log_p += init_ext_scorer_->beta;
            }
            // p_{nb}(l;x_{1:t})
            prefix_new->log_prob_nb_cur =
                log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
    }  // end of loop over prefix
    return 0;
}

void CTCBeamSearch::CalculateApproxScore() {
    size_t beam_size = opts_.beam_size;
    size_t num_prefixes_ = std::min(prefixes_.size(), beam_size);
    std::sort(
        prefixes_.begin(), prefixes_.begin() + num_prefixes_, prefix_compare);

    // compute aproximate ctc score as the return score, without affecting the
    // return order of decoding result. To delete when decoder gets stable.
    for (size_t i = 0; i < beam_size && i < prefixes_.size(); ++i) {
        double approx_ctc = prefixes_[i]->score;
        if (init_ext_scorer_ != nullptr) {
            vector<int> output;
            prefixes_[i]->get_path_vec(output);
            auto prefix_length = output.size();
            auto words = init_ext_scorer_->split_labels(output);
            // remove word insert
            approx_ctc = approx_ctc - prefix_length * init_ext_scorer_->beta;
            // remove language model weight:
            approx_ctc -= (init_ext_scorer_->get_sent_log_prob(words)) *
                          init_ext_scorer_->alpha;
        }
        prefixes_[i]->approx_ctc = approx_ctc;
    }
}

void CTCBeamSearch::LMRescore() {
    size_t beam_size = opts_.beam_size;
    if (init_ext_scorer_ != nullptr &&
        !init_ext_scorer_->is_character_based()) {
        for (size_t i = 0; i < beam_size && i < prefixes_.size(); ++i) {
            auto prefix = prefixes_[i];
            if (!prefix->is_empty() && prefix->character != space_id_) {
                float score = 0.0;
                vector<string> ngram = init_ext_scorer_->make_ngram(prefix);
                score = init_ext_scorer_->get_log_cond_prob(ngram) *
                        init_ext_scorer_->alpha;
                score += init_ext_scorer_->beta;
                prefix->score += score;
            }
        }
    }
}

}  // namespace ppspeech
