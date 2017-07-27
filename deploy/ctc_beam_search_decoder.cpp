#include <iostream>
#include <map>
#include <algorithm>
#include <utility>
#include <cmath>
#include <limits>
#include "ctc_beam_search_decoder.h"

typedef float log_prob_type;

template <typename T1, typename T2>
bool pair_comp_first_rev(const std::pair<T1, T2> a, const std::pair<T1, T2> b)
{
    return a.first > b.first;
}

template <typename T1, typename T2>
bool pair_comp_second_rev(const std::pair<T1, T2> a, const std::pair<T1, T2> b)
{
    return a.second > b.second;
}

template <typename T>
T log_sum_exp(T x, T y)
{
    static T num_min = -std::numeric_limits<T>::max();
    if (x <= -num_min) return y;
    if (y <= -num_min) return x;
    T xmax = std::max(x, y);
    return std::log(std::exp(x-xmax) + std::exp(y-xmax)) + xmax;
}

std::string ctc_best_path_decoder(std::vector<std::vector<double> > probs_seq,
                                  std::vector<std::string> vocabulary) {
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
        std::cout<<max_idx<<",";
        max_prob = 0.0;
        max_idx = 0;
    }
    std::cout<<std::endl;

    std::vector<int> idx_vec;
    for (int i=0; i<max_idx_vec.size(); i++) {
        std::cout<<max_idx_vec[i]<<",";
        if ((i == 0) || ((i>0) && max_idx_vec[i]!=max_idx_vec[i-1])) {
            std::cout<<max_idx_vec[i]<<",";
            idx_vec.push_back(max_idx_vec[i]);
        }
    }

    std::string best_path_result;
    for (int i=0; i<idx_vec.size(); i++) {
        if (idx_vec[i] != blank_id) {
            best_path_result += vocabulary[i];
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
                            Scorer *ext_scorer,
                            bool nproc) {
    // dimension check
    int num_time_steps = probs_seq.size();
    for (int i=0; i<num_time_steps; i++) {
        if (probs_seq[i].size() != vocabulary.size()+1) {
            std::cout<<"The shape of probs_seq does not match"
                     <<" with the shape of the vocabulary!"<<std::endl;
            exit(1);
        }
    }

    // blank_id check
    if (blank_id > vocabulary.size()) {
        std::cout<<"Invalid blank_id!"<<std::endl;
        exit(1);
    }

    // assign space ID
    std::vector<std::string>::iterator it = std::find(vocabulary.begin(),
                                                  vocabulary.end(), " ");
    int space_id = it - vocabulary.begin();
    if(space_id >= vocabulary.size()) {
        std::cout<<"The character space is not in the vocabulary!"<<std::endl;
        exit(1);
    }

    // initialize
    // two sets containing selected and candidate prefixes respectively
    std::map<std::string, log_prob_type> prefix_set_prev, prefix_set_next;
    // probability of prefixes ending with blank and non-blank
    std::map<std::string, log_prob_type> log_probs_b_prev, log_probs_nb_prev;
    std::map<std::string, log_prob_type> log_probs_b_cur, log_probs_nb_cur;

    static log_prob_type NUM_MAX = std::numeric_limits<log_prob_type>::max();
    prefix_set_prev["\t"] = 0.0;
    log_probs_b_prev["\t"] = 0.0;
    log_probs_nb_prev["\t"] = -NUM_MAX;

    for (int time_step=0; time_step<num_time_steps; time_step++) {
        prefix_set_next.clear();
        log_probs_b_cur.clear();
        log_probs_nb_cur.clear();
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
                        (prob_idx[i].first, log(prob_idx[i].second)));
        }

        // extend prefix
        for (std::map<std::string, log_prob_type>::iterator
             it = prefix_set_prev.begin();
            it != prefix_set_prev.end(); it++) {
            std::string l = it->first;
            if( prefix_set_next.find(l) == prefix_set_next.end()) {
                log_probs_b_cur[l] = log_probs_nb_cur[l] = -NUM_MAX;
            }

            for (int index=0; index<log_prob_idx.size(); index++) {
                int c = log_prob_idx[index].first;
                log_prob_type log_prob_c = log_prob_idx[index].second;
                log_prob_type log_probs_prev;
                if (c == blank_id) {
                    log_probs_prev = log_sum_exp(log_probs_b_prev[l],
                                                 log_probs_nb_prev[l]);
                    log_probs_b_cur[l] = log_sum_exp(log_probs_b_cur[l],
                                                     log_prob_c+log_probs_prev);
                } else {
                    std::string last_char = l.substr(l.size()-1, 1);
                    std::string new_char = vocabulary[c];
                    std::string l_plus = l + new_char;

                    if( prefix_set_next.find(l_plus) == prefix_set_next.end()) {
                        log_probs_b_cur[l_plus] = -NUM_MAX;
                        log_probs_nb_cur[l_plus] = -NUM_MAX;
                    }
                    if (last_char == new_char) {
                        log_probs_nb_cur[l_plus] = log_sum_exp(
                                                log_probs_nb_cur[l_plus],
                                                log_prob_c+log_probs_b_prev[l]
                                            );
                        log_probs_nb_cur[l] = log_sum_exp(
                                                log_probs_nb_cur[l],
                                                log_prob_c+log_probs_nb_prev[l]
                                            );
                    } else if (new_char == " ") {
                        float score = 0.0;
                        if (ext_scorer != NULL && l.size() > 1) {
                            score = ext_scorer->get_score(l.substr(1), true);
                        }
                        log_probs_prev = log_sum_exp(log_probs_b_prev[l],
                                                     log_probs_nb_prev[l]);
                        log_probs_nb_cur[l_plus] = log_sum_exp(
                                            log_probs_nb_cur[l_plus],
                                            score + log_prob_c + log_probs_prev
                                        );
                    } else {
                        log_probs_prev = log_sum_exp(log_probs_b_prev[l],
                                                     log_probs_nb_prev[l]);
                        log_probs_nb_cur[l_plus] = log_sum_exp(
                                                    log_probs_nb_cur[l_plus],
                                                    log_prob_c+log_probs_prev
                                                );
                    }
                    prefix_set_next[l_plus] = log_sum_exp(
                                                log_probs_nb_cur[l_plus],
                                                log_probs_b_cur[l_plus]
                                            );
                }
            }

            prefix_set_next[l] = log_sum_exp(log_probs_b_cur[l],
                                             log_probs_nb_cur[l]);
        }

        log_probs_b_prev = log_probs_b_cur;
        log_probs_nb_prev = log_probs_nb_cur;
        std::vector<std::pair<std::string, log_prob_type> >
                  prefix_vec_next(prefix_set_next.begin(),
                                  prefix_set_next.end());
        std::sort(prefix_vec_next.begin(),
                  prefix_vec_next.end(),
                  pair_comp_second_rev<std::string, log_prob_type>);
        int num_prefixes_next = prefix_vec_next.size();
        int k = beam_size<num_prefixes_next ? beam_size : num_prefixes_next;
        prefix_set_prev = std::map<std::string, log_prob_type> (
                                                   prefix_vec_next.begin(),
                                                   prefix_vec_next.begin() + k
                                                );
    }

    // post processing
    std::vector<std::pair<double, std::string> > beam_result;
    for (std::map<std::string, log_prob_type>::iterator
         it = prefix_set_prev.begin(); it != prefix_set_prev.end(); it++) {
        if (it->second > -NUM_MAX && it->first.size() > 1) {
            log_prob_type log_prob = it->second;
            std::string sentence = it->first.substr(1);
            // scoring the last word
            if (ext_scorer != NULL && sentence[sentence.size()-1] != ' ') {
                log_prob = log_prob + ext_scorer->get_score(sentence, true);
            }
            if (log_prob > -NUM_MAX) {
                std::pair<double, std::string> cur_result(log_prob, sentence);
                beam_result.push_back(cur_result);
            }
        }
    }
    // sort the result and return
    std::sort(beam_result.begin(), beam_result.end(),
              pair_comp_first_rev<double, std::string>);
    return beam_result;
}
