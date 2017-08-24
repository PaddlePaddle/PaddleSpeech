#include <iostream>
#include <map>
#include <algorithm>
#include <utility>
#include <cmath>
#include <limits>
#include "fst/fstlib.h"
#include "ctc_decoders.h"
#include "decoder_utils.h"
#include "path_trie.h"
#include "ThreadPool.h"

typedef float log_prob_type;

std::string ctc_best_path_decoder(std::vector<std::vector<double> > probs_seq,
                                  std::vector<std::string> vocabulary)
{
    // dimension check
    int num_time_steps = probs_seq.size();
    for (int i=0; i<num_time_steps; i++) {
        if (probs_seq[i].size() != vocabulary.size()+1) {
            std::cout<<"The shape of probs_seq does not match"
                     <<" with the shape of the vocabulary!"<<std::endl;
            exit(1);
        }
    }

    int blank_id = vocabulary.size();

    std::vector<int> max_idx_vec;
    double max_prob = 0.0;
    int max_idx = 0;
    for (int i=0; i<num_time_steps; i++) {
        for (int j=0; j<probs_seq[i].size(); j++) {
            if (max_prob < probs_seq[i][j]) {
                max_idx = j;
                max_prob = probs_seq[i][j];
            }
        }
        max_idx_vec.push_back(max_idx);
        max_prob = 0.0;
        max_idx = 0;
    }

    std::vector<int> idx_vec;
    for (int i=0; i<max_idx_vec.size(); i++) {
        if ((i == 0) || ((i>0) && max_idx_vec[i]!=max_idx_vec[i-1])) {
            idx_vec.push_back(max_idx_vec[i]);
        }
    }

    std::string best_path_result;
    for (int i=0; i<idx_vec.size(); i++) {
        if (idx_vec[i] != blank_id) {
            best_path_result += vocabulary[idx_vec[i]];
        }
    }
    return best_path_result;
}

std::vector<std::pair<double, std::string> >
    ctc_beam_search_decoder(std::vector<std::vector<double> > probs_seq,
                            int beam_size,
                            std::vector<std::string> vocabulary,
                            int blank_id,
                            double cutoff_prob,
                            Scorer *ext_scorer)
{
    // dimension check
    int num_time_steps = probs_seq.size();
    for (int i=0; i<num_time_steps; i++) {
        if (probs_seq[i].size() != vocabulary.size()+1) {
            std::cout << " The shape of probs_seq does not match"
                      << " with the shape of the vocabulary!" << std::endl;
            exit(1);
        }
    }

    // blank_id check
    if (blank_id > vocabulary.size()) {
        std::cout << " Invalid blank_id! " << std::endl;
        exit(1);
    }

    // assign space ID
    std::vector<std::string>::iterator it = std::find(vocabulary.begin(),
                                                  vocabulary.end(), " ");
    int space_id = it - vocabulary.begin();
    if(space_id >= vocabulary.size()) {
        std::cout << " The character space is not in the vocabulary!"<<std::endl;
        exit(1);
    }

    static log_prob_type POS_INF = std::numeric_limits<log_prob_type>::max();
    static log_prob_type NEG_INF = -POS_INF;
    static log_prob_type NUM_MIN = std::numeric_limits<log_prob_type>::min();

    // init
    PathTrie root;
    root._log_prob_b_prev = 0.0;
    root._score = 0.0;
    std::vector<PathTrie*> prefixes;
    prefixes.push_back(&root);

    if ( ext_scorer != nullptr && !ext_scorer->is_character_based()) {
        if (ext_scorer->dictionary == nullptr) {
        // TODO: init dictionary
        }
        auto fst_dict = static_cast<fst::StdVectorFst*>(ext_scorer->dictionary);
        fst::StdVectorFst* dict_ptr = fst_dict->Copy(true);
        root.set_dictionary(dict_ptr);
        auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
        root.set_matcher(matcher);
    }

    for (int time_step = 0; time_step < num_time_steps; time_step++) {
        std::vector<double> prob = probs_seq[time_step];
        std::vector<std::pair<int, double> > prob_idx;
        for (int i=0; i<prob.size(); i++) {
            prob_idx.push_back(std::pair<int, double>(i, prob[i]));
        }

        // pruning of vacobulary
        int cutoff_len = prob.size();
        if (cutoff_prob < 1.0) {
            std::sort(prob_idx.begin(),
                      prob_idx.end(),
                      pair_comp_second_rev<int, double>);
            double cum_prob = 0.0;
            cutoff_len = 0;
            for (int i=0; i<prob_idx.size(); i++) {
                cum_prob += prob_idx[i].second;
                cutoff_len += 1;
                if (cum_prob >= cutoff_prob) break;
            }
            prob_idx = std::vector<std::pair<int, double> >( prob_idx.begin(),
                            prob_idx.begin() + cutoff_len);
        }

        std::vector<std::pair<int, log_prob_type> > log_prob_idx;
        for (int i=0; i<cutoff_len; i++) {
            log_prob_idx.push_back(std::pair<int, log_prob_type>
                        (prob_idx[i].first, log(prob_idx[i].second + NUM_MIN)));
        }

        // loop over chars
        for (int index = 0; index < log_prob_idx.size(); index++) {
            auto c = log_prob_idx[index].first;
            log_prob_type log_prob_c = log_prob_idx[index].second;
            //log_prob_type log_probs_prev;

            for (int i = 0; i < prefixes.size() && i<beam_size; i++) {
                auto prefix = prefixes[i];
                // blank
                if (c == blank_id) {
                    prefix->_log_prob_b_cur = log_sum_exp(
                                               prefix->_log_prob_b_cur,
                                               log_prob_c + prefix->_score);
                    continue;
                }
                // repeated character
                if (c == prefix->_character) {
                    prefix->_log_prob_nb_cur = log_sum_exp(
                        prefix->_log_prob_nb_cur,
                        log_prob_c + prefix->_log_prob_nb_prev
                        );
                }
                // get new prefix
                auto prefix_new = prefix->get_path_trie(c);

                if (prefix_new != nullptr) {
                    float log_p = NEG_INF;

                    if (c == prefix->_character
                        && prefix->_log_prob_b_prev > NEG_INF) {
                        log_p = log_prob_c + prefix->_log_prob_b_prev;
                    } else if (c != prefix->_character) {
                        log_p = log_prob_c + prefix->_score;
                    }

                    // language model scoring
                    if (ext_scorer != nullptr &&
                        (c == space_id || ext_scorer->is_character_based()) ) {
                        PathTrie *prefix_to_score = nullptr;

                        // don't score the space
                        if (ext_scorer->is_character_based()) {
                            prefix_to_score = prefix_new;
                        } else {
                            prefix_to_score = prefix;
                        }

                        double score = 0.0;
                        std::vector<std::string> ngram;
                        ngram = ext_scorer->make_ngram(prefix_to_score);
                        score = ext_scorer->get_log_cond_prob(ngram) *
                                ext_scorer->alpha;

                        log_p += score;
                        log_p += ext_scorer->beta;

                    }
                    prefix_new->_log_prob_nb_cur = log_sum_exp(
                                        prefix_new->_log_prob_nb_cur, log_p);
                }
            }

        } // end of loop over chars

        prefixes.clear();
        // update log probabilities
        root.iterate_to_vec(prefixes);

        // sort prefixes by score
        if (prefixes.size() >= beam_size) {
            std::nth_element(prefixes.begin(),
                    prefixes.begin() + beam_size,
                    prefixes.end(),
                    prefix_compare);

            for (size_t i = beam_size; i < prefixes.size(); i++) {
                prefixes[i]->remove();
            }
        }
    }

    for (size_t i = 0; i < beam_size && i < prefixes.size(); i++) {
        double approx_ctc = prefixes[i]->_score;

        // remove word insert:
        std::vector<int> output;
        prefixes[i]->get_path_vec(output);
        size_t prefix_length = output.size();
        // remove language model weight:
        if (ext_scorer != nullptr) {
           // auto words = split_labels(output);
           // approx_ctc = approx_ctc - path_length * ext_scorer->beta;
           // approx_ctc -= (_lm->get_sent_log_prob(words)) * ext_scorer->alpha;
        }

        prefixes[i]->_approx_ctc = approx_ctc;
    }

    // allow for the post processing
    std::vector<PathTrie*> space_prefixes;
    if (space_prefixes.empty()) {
        for (size_t i = 0; i < beam_size && i< prefixes.size(); i++) {
            space_prefixes.push_back(prefixes[i]);
        }
    }

    std::sort(space_prefixes.begin(), space_prefixes.end(), prefix_compare);
    std::vector<std::pair<double, std::string> > output_vecs;
    for (size_t i = 0; i < beam_size && i < space_prefixes.size(); i++) {
        std::vector<int> output;
        space_prefixes[i]->get_path_vec(output);
        // convert index to string
        std::string output_str;
        for (int j = 0; j < output.size(); j++) {
            output_str += vocabulary[output[j]];
        }
        std::pair<double, std::string> output_pair(space_prefixes[i]->_score,
                                                   output_str);
        output_vecs.emplace_back(
            output_pair
        );
    }

    return output_vecs;
 }


std::vector<std::vector<std::pair<double, std::string>>>
    ctc_beam_search_decoder_batch(
                std::vector<std::vector<std::vector<double>>> probs_split,
                int beam_size,
                std::vector<std::string> vocabulary,
                int blank_id,
                int num_processes,
                double cutoff_prob,
                Scorer *ext_scorer
                ) {
    if (num_processes <= 0) {
        std::cout << "num_processes must be nonnegative!" << std::endl;
        exit(1);
    }
    // thread pool
    ThreadPool pool(num_processes);
    // number of samples
    int batch_size = probs_split.size();
    // enqueue the tasks of decoding
    std::vector<std::future<std::vector<std::pair<double, std::string>>>> res;
    for (int i = 0; i < batch_size; i++) {
        res.emplace_back(
                pool.enqueue(ctc_beam_search_decoder, probs_split[i],
                    beam_size, vocabulary, blank_id, cutoff_prob, ext_scorer)
            );
    }
    // get decoding results
    std::vector<std::vector<std::pair<double, std::string>>> batch_results;
    for (int i = 0; i < batch_size; i++) {
        batch_results.emplace_back(res[i].get());
    }
    return batch_results;
}
