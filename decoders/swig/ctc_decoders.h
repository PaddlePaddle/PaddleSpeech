#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <string>
#include <utility>
#include <vector>

#include "scorer.h"

/* CTC Best Path Decoder
 *
 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary: A vector of vocabulary.
 * Return:
 *     The decoding result in string
 */
std::string ctc_greedy_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary);

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     beam_size: The width of beam search.
 *     vocabulary: A vector of vocabulary.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/
std::vector<std::pair<double, std::string>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const size_t beam_size,
    std::vector<std::string> vocabulary,
    const double cutoff_prob = 1.0,
    const size_t cutoff_top_n = 40,
    Scorer *ext_scorer = NULL);

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *      .
 *     beam_size: The width of beam search.
 *     vocabulary: A vector of vocabulary.
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
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const size_t beam_size,
    const std::vector<std::string> &vocabulary,
    const size_t num_processes,
    double cutoff_prob = 1.0,
    const size_t cutoff_top_n = 40,
    Scorer *ext_scorer = NULL);

#endif  // CTC_BEAM_SEARCH_DECODER_H_
