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

#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <string>
#include <utility>
#include <vector>

#include "scorer.h"

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/
std::vector<std::pair<double, std::string>> ctc_beam_search_decoding(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    Scorer *ext_scorer = nullptr,
    size_t blank_id = 0);


/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<double, std::string>>>
ctc_beam_search_decoding_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    Scorer *ext_scorer = nullptr,
    size_t blank_id = 0);

/**
 * Store the root and prefixes for decoder
 */

class CtcBeamSearchDecoderStorage {
  public:
    PathTrie *root = nullptr;
    std::vector<PathTrie *> prefixes;

    CtcBeamSearchDecoderStorage() {
        // init prefixes' root
        this->root = new PathTrie();
        this->root->log_prob_b_prev = 0.0;
        // The score of root is in log scale.Since the prob=1.0, the prob score
        // in log scale is 0.0
        this->root->score = root->log_prob_b_prev;
        // std::vector<PathTrie *> prefixes;
        this->prefixes.push_back(root);
    };

    ~CtcBeamSearchDecoderStorage() {
        if (root != nullptr) {
            delete root;
            root = nullptr;
        }
    };
};

/**
 * The ctc beam search decoder, support batchsize >= 1
 */
class CtcBeamSearchDecoderBatch {
  public:
    CtcBeamSearchDecoderBatch(const std::vector<std::string> &vocabulary,
                              size_t batch_size,
                              size_t beam_size,
                              size_t num_processes,
                              double cutoff_prob,
                              size_t cutoff_top_n,
                              Scorer *ext_scorer,
                              size_t blank_id);

    ~CtcBeamSearchDecoderBatch();
    void next(const std::vector<std::vector<std::vector<double>>> &probs_split,
              const std::vector<std::string> &has_value);

    std::vector<std::vector<std::pair<double, std::string>>> decode();

    void reset_state(size_t batch_size,
                     size_t beam_size,
                     size_t num_processes,
                     double cutoff_prob,
                     size_t cutoff_top_n);

  private:
    std::vector<std::string> vocabulary;
    size_t batch_size;
    size_t beam_size;
    size_t num_processes;
    double cutoff_prob;
    size_t cutoff_top_n;
    Scorer *ext_scorer;
    size_t blank_id;
    std::vector<std::unique_ptr<CtcBeamSearchDecoderStorage>>
        decoder_storage_vector;
};

/**
 * function for chunk decoding
 */
void ctc_beam_search_decode_chunk(
    PathTrie *root,
    std::vector<PathTrie *> &prefixes,
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer,
    size_t blank_id);

std::vector<std::pair<double, std::string>> get_decode_result(
    std::vector<PathTrie *> &prefixes,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    Scorer *ext_scorer);

/**
 * free the CtcBeamSearchDecoderStorage
 */
void free_storage(std::unique_ptr<CtcBeamSearchDecoderStorage> &storage);

/**
 * initialize the root
 */
void ctc_beam_search_decode_chunk_begin(PathTrie *root, Scorer *ext_scorer);

#endif  // CTC_BEAM_SEARCH_DECODER_H_
