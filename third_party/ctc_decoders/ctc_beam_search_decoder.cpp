// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "COPYING.APACHE2.0");
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

#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "ThreadPool.h"
#include "fst/fstlib.h"

#include "decoder_utils.h"
#include "path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;


std::vector<std::pair<double, std::string>> ctc_beam_search_decoding(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer,
    size_t blank_id) {
    // dimension check
    size_t num_time_steps = probs_seq.size();
    for (size_t i = 0; i < num_time_steps; ++i) {
        VALID_CHECK_EQ(probs_seq[i].size(),
                       // vocabulary.size() + 1,
                       vocabulary.size(),
                       "The shape of probs_seq does not match with "
                       "the shape of the vocabulary");
    }


    // assign space id
    auto it = std::find(vocabulary.begin(), vocabulary.end(), kSPACE);
    int space_id = it - vocabulary.begin();
    // if no space in vocabulary
    if ((size_t)space_id >= vocabulary.size()) {
        space_id = -2;
    }
    // init prefixes' root
    PathTrie root;
    root.score = root.log_prob_b_prev = 0.0;
    std::vector<PathTrie *> prefixes;
    prefixes.push_back(&root);

    if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
        auto fst_dict =
            static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
        fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
        root.set_dictionary(dict_ptr);
        auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
        root.set_matcher(matcher);
    }

    // prefix search over time
    for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
        auto &prob = probs_seq[time_step];

        float min_cutoff = -NUM_FLT_INF;
        bool full_beam = false;
        if (ext_scorer != nullptr) {
            size_t num_prefixes = std::min(prefixes.size(), beam_size);
            std::sort(prefixes.begin(),
                      prefixes.begin() + num_prefixes,
                      prefix_compare);
            min_cutoff = prefixes[num_prefixes - 1]->score +
                         std::log(prob[blank_id]) -
                         std::max(0.0, ext_scorer->beta);
            full_beam = (num_prefixes == beam_size);
        }

        std::vector<std::pair<size_t, float>> log_prob_idx =
            get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n);
        // loop over chars
        for (size_t index = 0; index < log_prob_idx.size(); index++) {
            auto c = log_prob_idx[index].first;
            auto log_prob_c = log_prob_idx[index].second;

            for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
                auto prefix = prefixes[i];
                if (full_beam && log_prob_c + prefix->score < min_cutoff) {
                    break;
                }
                // blank
                if (c == blank_id) {
                    prefix->log_prob_b_cur = log_sum_exp(
                        prefix->log_prob_b_cur, log_prob_c + prefix->score);
                    continue;
                }
                // repeated character
                if (c == prefix->character) {
                    prefix->log_prob_nb_cur =
                        log_sum_exp(prefix->log_prob_nb_cur,
                                    log_prob_c + prefix->log_prob_nb_prev);
                }
                // get new prefix
                auto prefix_new = prefix->get_path_trie(c);

                if (prefix_new != nullptr) {
                    float log_p = -NUM_FLT_INF;

                    if (c == prefix->character &&
                        prefix->log_prob_b_prev > -NUM_FLT_INF) {
                        log_p = log_prob_c + prefix->log_prob_b_prev;
                    } else if (c != prefix->character) {
                        log_p = log_prob_c + prefix->score;
                    }

                    // language model scoring
                    if (ext_scorer != nullptr &&
                        (c == space_id || ext_scorer->is_character_based())) {
                        PathTrie *prefix_to_score = nullptr;
                        // skip scoring the space
                        if (ext_scorer->is_character_based()) {
                            prefix_to_score = prefix_new;
                        } else {
                            prefix_to_score = prefix;
                        }

                        float score = 0.0;
                        std::vector<std::string> ngram;
                        ngram = ext_scorer->make_ngram(prefix_to_score);
                        score = ext_scorer->get_log_cond_prob(ngram) *
                                ext_scorer->alpha;
                        log_p += score;
                        log_p += ext_scorer->beta;
                    }
                    prefix_new->log_prob_nb_cur =
                        log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
                }
            }  // end of loop over prefix
        }      // end of loop over vocabulary


        prefixes.clear();
        // update log probs
        root.iterate_to_vec(prefixes);

        // only preserve top beam_size prefixes
        if (prefixes.size() >= beam_size) {
            std::nth_element(prefixes.begin(),
                             prefixes.begin() + beam_size,
                             prefixes.end(),
                             prefix_compare);
            for (size_t i = beam_size; i < prefixes.size(); ++i) {
                prefixes[i]->remove();
            }
        }
    }  // end of loop over time

    // score the last word of each prefix that doesn't end with space
    if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
        for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
            auto prefix = prefixes[i];
            if (!prefix->is_empty() && prefix->character != space_id) {
                float score = 0.0;
                std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
                score =
                    ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
                score += ext_scorer->beta;
                prefix->score += score;
            }
        }
    }

    size_t num_prefixes = std::min(prefixes.size(), beam_size);
    std::sort(
        prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);

    // compute approximate ctc score as the return score, without affecting the
    // return order of decoding result. To delete when decoder gets stable.
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
        double approx_ctc = prefixes[i]->score;
        if (ext_scorer != nullptr) {
            std::vector<int> output;
            prefixes[i]->get_path_vec(output);
            auto prefix_length = output.size();
            auto words = ext_scorer->split_labels(output);
            // remove word insert
            approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
            // remove language model weight:
            approx_ctc -=
                (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
        }
        prefixes[i]->approx_ctc = approx_ctc;
    }

    return get_beam_search_result(prefixes, vocabulary, beam_size);
}


std::vector<std::vector<std::pair<double, std::string>>>
ctc_beam_search_decoding_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer,
    size_t blank_id) {
    VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
    // thread pool
    ThreadPool pool(num_processes);
    // number of samples
    size_t batch_size = probs_split.size();

    // enqueue the tasks of decoding
    std::vector<std::future<std::vector<std::pair<double, std::string>>>> res;
    for (size_t i = 0; i < batch_size; ++i) {
        res.emplace_back(pool.enqueue(ctc_beam_search_decoding,
                                      probs_split[i],
                                      vocabulary,
                                      beam_size,
                                      cutoff_prob,
                                      cutoff_top_n,
                                      ext_scorer,
                                      blank_id));
    }

    // get decoding results
    std::vector<std::vector<std::pair<double, std::string>>> batch_results;
    for (size_t i = 0; i < batch_size; ++i) {
        batch_results.emplace_back(res[i].get());
    }
    return batch_results;
}

void ctc_beam_search_decode_chunk_begin(PathTrie *root, Scorer *ext_scorer) {
    if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
        auto fst_dict =
            static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
        fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
        root->set_dictionary(dict_ptr);
        auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
        root->set_matcher(matcher);
    }
}

void ctc_beam_search_decode_chunk(
    PathTrie *root,
    std::vector<PathTrie *> &prefixes,
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer,
    size_t blank_id) {
    // dimension check
    size_t num_time_steps = probs_seq.size();
    for (size_t i = 0; i < num_time_steps; ++i) {
        VALID_CHECK_EQ(probs_seq[i].size(),
                       // vocabulary.size() + 1,
                       vocabulary.size(),
                       "The shape of probs_seq does not match with "
                       "the shape of the vocabulary");
    }

    // assign space id
    auto it = std::find(vocabulary.begin(), vocabulary.end(), kSPACE);
    int space_id = it - vocabulary.begin();
    // if no space in vocabulary
    if ((size_t)space_id >= vocabulary.size()) {
        space_id = -2;
    }
    // init prefixes' root
    //
    // prefix search over time
    for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
        auto &prob = probs_seq[time_step];

        float min_cutoff = -NUM_FLT_INF;
        bool full_beam = false;
        if (ext_scorer != nullptr) {
            size_t num_prefixes = std::min(prefixes.size(), beam_size);
            std::sort(prefixes.begin(),
                      prefixes.begin() + num_prefixes,
                      prefix_compare);
            min_cutoff = prefixes[num_prefixes - 1]->score +
                         std::log(prob[blank_id]) -
                         std::max(0.0, ext_scorer->beta);
            full_beam = (num_prefixes == beam_size);
        }

        std::vector<std::pair<size_t, float>> log_prob_idx =
            get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n);
        // loop over chars
        for (size_t index = 0; index < log_prob_idx.size(); index++) {
            auto c = log_prob_idx[index].first;
            auto log_prob_c = log_prob_idx[index].second;

            for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
                auto prefix = prefixes[i];
                if (full_beam && log_prob_c + prefix->score < min_cutoff) {
                    break;
                }
                // blank
                if (c == blank_id) {
                    prefix->log_prob_b_cur = log_sum_exp(
                        prefix->log_prob_b_cur, log_prob_c + prefix->score);
                    continue;
                }
                // repeated character
                if (c == prefix->character) {
                    prefix->log_prob_nb_cur =
                        log_sum_exp(prefix->log_prob_nb_cur,
                                    log_prob_c + prefix->log_prob_nb_prev);
                }
                // get new prefix
                auto prefix_new = prefix->get_path_trie(c);

                if (prefix_new != nullptr) {
                    float log_p = -NUM_FLT_INF;

                    if (c == prefix->character &&
                        prefix->log_prob_b_prev > -NUM_FLT_INF) {
                        log_p = log_prob_c + prefix->log_prob_b_prev;
                    } else if (c != prefix->character) {
                        log_p = log_prob_c + prefix->score;
                    }

                    // language model scoring
                    if (ext_scorer != nullptr &&
                        (c == space_id || ext_scorer->is_character_based())) {
                        PathTrie *prefix_to_score = nullptr;
                        // skip scoring the space
                        if (ext_scorer->is_character_based()) {
                            prefix_to_score = prefix_new;
                        } else {
                            prefix_to_score = prefix;
                        }

                        float score = 0.0;
                        std::vector<std::string> ngram;
                        ngram = ext_scorer->make_ngram(prefix_to_score);
                        score = ext_scorer->get_log_cond_prob(ngram) *
                                ext_scorer->alpha;
                        log_p += score;
                        log_p += ext_scorer->beta;
                    }
                    prefix_new->log_prob_nb_cur =
                        log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
                }
            }  // end of loop over prefix
        }      // end of loop over vocabulary

        prefixes.clear();
        // update log probs

        root->iterate_to_vec(prefixes);

        // only preserve top beam_size prefixes
        if (prefixes.size() >= beam_size) {
            std::nth_element(prefixes.begin(),
                             prefixes.begin() + beam_size,
                             prefixes.end(),
                             prefix_compare);
            for (size_t i = beam_size; i < prefixes.size(); ++i) {
                prefixes[i]->remove();
            }
        }
    }  // end of loop over time

    return;
}


std::vector<std::pair<double, std::string>> get_decode_result(
    std::vector<PathTrie *> &prefixes,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    Scorer *ext_scorer) {
    auto it = std::find(vocabulary.begin(), vocabulary.end(), kSPACE);
    int space_id = it - vocabulary.begin();
    // if no space in vocabulary
    if ((size_t)space_id >= vocabulary.size()) {
        space_id = -2;
    }
    // score the last word of each prefix that doesn't end with space
    if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
        for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
            auto prefix = prefixes[i];
            if (!prefix->is_empty() && prefix->character != space_id) {
                float score = 0.0;
                std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
                score =
                    ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
                score += ext_scorer->beta;
                prefix->score += score;
            }
        }
    }

    size_t num_prefixes = std::min(prefixes.size(), beam_size);
    std::sort(
        prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);

    // compute aproximate ctc score as the return score, without affecting the
    // return order of decoding result. To delete when decoder gets stable.
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
        double approx_ctc = prefixes[i]->score;
        if (ext_scorer != nullptr) {
            std::vector<int> output;
            prefixes[i]->get_path_vec(output);
            auto prefix_length = output.size();
            auto words = ext_scorer->split_labels(output);
            // remove word insert
            approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
            // remove language model weight:
            approx_ctc -=
                (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
        }
        prefixes[i]->approx_ctc = approx_ctc;
    }

    std::vector<std::pair<double, std::string>> res =
        get_beam_search_result(prefixes, vocabulary, beam_size);

    // pay back the last word of each prefix that doesn't end with space (for
    // decoding by chunk)
    if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
        for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
            auto prefix = prefixes[i];
            if (!prefix->is_empty() && prefix->character != space_id) {
                float score = 0.0;
                std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
                score =
                    ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
                score += ext_scorer->beta;
                prefix->score -= score;
            }
        }
    }
    return res;
}


void free_storage(std::unique_ptr<CtcBeamSearchDecoderStorage> &storage) {
    storage = nullptr;
}


CtcBeamSearchDecoderBatch::~CtcBeamSearchDecoderBatch() {}

CtcBeamSearchDecoderBatch::CtcBeamSearchDecoderBatch(
    const std::vector<std::string> &vocabulary,
    size_t batch_size,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer,
    size_t blank_id)
    : batch_size(batch_size),
      beam_size(beam_size),
      num_processes(num_processes),
      cutoff_prob(cutoff_prob),
      cutoff_top_n(cutoff_top_n),
      ext_scorer(ext_scorer),
      blank_id(blank_id) {
    VALID_CHECK_GT(this->beam_size, 0, "beam_size must be greater than 0!");
    VALID_CHECK_GT(
        this->num_processes, 0, "num_processes must be nonnegative!");
    this->vocabulary = vocabulary;
    for (size_t i = 0; i < batch_size; i++) {
        this->decoder_storage_vector.push_back(
            std::unique_ptr<CtcBeamSearchDecoderStorage>(
                new CtcBeamSearchDecoderStorage()));
        ctc_beam_search_decode_chunk_begin(
            this->decoder_storage_vector[i]->root, ext_scorer);
    }
};

/**
 * Input
 * probs_split: shape [B, T, D]
 */
void CtcBeamSearchDecoderBatch::next(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &has_value) {
    VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
    // thread pool
    size_t num_has_value = 0;
    for (int i = 0; i < has_value.size(); i++)
        if (has_value[i] == "true") num_has_value += 1;
    ThreadPool pool(std::min(num_processes, num_has_value));
    // number of samples
    size_t probs_num = probs_split.size();
    VALID_CHECK_EQ(this->batch_size,
                   probs_num,
                   "The batch size of the current input data should be same "
                   "with the input data before");

    // enqueue the tasks of decoding
    std::vector<std::future<void>> res;
    for (size_t i = 0; i < batch_size; ++i) {
        if (has_value[i] == "true") {
            res.emplace_back(pool.enqueue(
                ctc_beam_search_decode_chunk,
                std::ref(this->decoder_storage_vector[i]->root),
                std::ref(this->decoder_storage_vector[i]->prefixes),
                probs_split[i],
                this->vocabulary,
                this->beam_size,
                this->cutoff_prob,
                this->cutoff_top_n,
                this->ext_scorer,
                this->blank_id));
        }
    }

    for (size_t i = 0; i < batch_size; ++i) {
        res[i].get();
    }
    return;
};

/**
 * Return
 * batch_result: shape[B, beam_size,(-approx_ctc score, string)]
 */
std::vector<std::vector<std::pair<double, std::string>>>
CtcBeamSearchDecoderBatch::decode() {
    VALID_CHECK_GT(
        this->num_processes, 0, "num_processes must be nonnegative!");
    // thread pool
    ThreadPool pool(this->num_processes);
    // number of samples
    // enqueue the tasks of decoding
    std::vector<std::future<std::vector<std::pair<double, std::string>>>> res;
    for (size_t i = 0; i < this->batch_size; ++i) {
        res.emplace_back(
            pool.enqueue(get_decode_result,
                         std::ref(this->decoder_storage_vector[i]->prefixes),
                         this->vocabulary,
                         this->beam_size,
                         this->ext_scorer));
    }
    // get decoding results
    std::vector<std::vector<std::pair<double, std::string>>> batch_results;
    for (size_t i = 0; i < this->batch_size; ++i) {
        batch_results.emplace_back(res[i].get());
    }
    return batch_results;
}


/**
 * reset the state of ctcBeamSearchDecoderBatch
 */
void CtcBeamSearchDecoderBatch::reset_state(size_t batch_size,
                                            size_t beam_size,
                                            size_t num_processes,
                                            double cutoff_prob,
                                            size_t cutoff_top_n) {
    this->batch_size = batch_size;
    this->beam_size = beam_size;
    this->num_processes = num_processes;
    this->cutoff_prob = cutoff_prob;
    this->cutoff_top_n = cutoff_top_n;

    VALID_CHECK_GT(this->beam_size, 0, "beam_size must be greater than 0!");
    VALID_CHECK_GT(
        this->num_processes, 0, "num_processes must be nonnegative!");
    // thread pool
    ThreadPool pool(this->num_processes);
    // number of samples
    // enqueue the tasks of decoding
    std::vector<std::future<void>> res;
    size_t storage_size = decoder_storage_vector.size();
    for (size_t i = 0; i < storage_size; i++) {
        res.emplace_back(pool.enqueue(
            free_storage, std::ref(this->decoder_storage_vector[i])));
    }
    for (size_t i = 0; i < storage_size; ++i) {
        res[i].get();
    }
    std::vector<std::unique_ptr<CtcBeamSearchDecoderStorage>>().swap(
        decoder_storage_vector);
    for (size_t i = 0; i < this->batch_size; i++) {
        this->decoder_storage_vector.push_back(
            std::unique_ptr<CtcBeamSearchDecoderStorage>(
                new CtcBeamSearchDecoderStorage()));
        ctc_beam_search_decode_chunk_begin(
            this->decoder_storage_vector[i]->root, this->ext_scorer);
    }
}