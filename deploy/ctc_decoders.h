#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <vector>
#include <string>
#include <utility>
#include "scorer.h"

/* CTC Best Path Decoder
 *
 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary: A vector of vocabulary.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
 */
std::string ctc_best_path_decoder(std::vector<std::vector<double> > probs_seq,
                                     std::vector<std::string> vocabulary);

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     beam_size: The width of beam search.
 *     vocabulary: A vector of vocabulary.
 *     blank_id: ID of blank.
 *     cutoff_prob: Cutoff probability of pruning
 *     ext_scorer: External scorer to evaluate a prefix.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/
std::vector<std::pair<double, std::string> >
    ctc_beam_search_decoder(std::vector<std::vector<double> > probs_seq,
                            int beam_size,
                            std::vector<std::string> vocabulary,
                            int blank_id,
                            double cutoff_prob=1.0,
                            int cutoff_top_n=40,
                            Scorer *ext_scorer=NULL
                            );

/* CTC Beam Search Decoder for batch data, the interface is consistent with the
 * original decoder in Python version.

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *      .
 *     beam_size: The width of beam search.
 *     vocabulary: A vector of vocabulary.
 *     blank_id: ID of blank.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability of pruning
 *     ext_scorer: External scorer to evaluate a prefix.
 * Return:
 *     A 2-D vector that each element is a vector of decoding result for one
 *     sample.
*/
std::vector<std::vector<std::pair<double, std::string>>>
    ctc_beam_search_decoder_batch(std::vector<std::vector<std::vector<double>>> probs_split,
                            int beam_size,
                            std::vector<std::string> vocabulary,
                            int blank_id,
                            int num_processes,
                            double cutoff_prob=1.0,
                            int cutoff_top_n=40,
                            Scorer *ext_scorer=NULL
                            );


#endif // CTC_BEAM_SEARCH_DECODER_H_
