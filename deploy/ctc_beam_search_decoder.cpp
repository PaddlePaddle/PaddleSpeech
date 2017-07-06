#include <iostream>
#include <map>
#include <algorithm>
#include <utility>
#include <cmath>
#include "ctc_beam_search_decoder.h"

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
    std::map<std::string, double> prefix_set_prev, prefix_set_next;
    // probability of prefixes ending with blank and non-blank
    std::map<std::string, double> probs_b_prev, probs_nb_prev;
    std::map<std::string, double> probs_b_cur, probs_nb_cur;
    prefix_set_prev["\t"] = 1.0;
    probs_b_prev["\t"] = 1.0;
    probs_nb_prev["\t"] = 0.0;

    for (int time_step=0; time_step<num_time_steps; time_step++) {
        prefix_set_next.clear();
        probs_b_cur.clear();
        probs_nb_cur.clear();
        std::vector<double> prob = probs_seq[time_step];

        std::vector<std::pair<int, double> > prob_idx;
        for (int i=0; i<prob.size(); i++) {
            prob_idx.push_back(std::pair<int, double>(i, prob[i]));
        }
        // pruning of vacobulary
        if (cutoff_prob < 1.0) {
            std::sort(prob_idx.begin(), prob_idx.end(),
                      pair_comp_second_rev<int, double>);
            float cum_prob = 0.0;
            int cutoff_len = 0;
            for (int i=0; i<prob_idx.size(); i++) {
                cum_prob += prob_idx[i].second;
                cutoff_len += 1;
                if (cum_prob >= cutoff_prob) break;
            }
            prob_idx = std::vector<std::pair<int, double> >( prob_idx.begin(),
                                                  prob_idx.begin() + cutoff_len);
        }
        // extend prefix
        for (std::map<std::string, double>::iterator it = prefix_set_prev.begin();
            it != prefix_set_prev.end(); it++) {
            std::string l = it->first;
            if( prefix_set_next.find(l) == prefix_set_next.end()) {
                probs_b_cur[l] = probs_nb_cur[l] = 0.0;
            }

            for (int index=0; index<prob_idx.size(); index++) {
                int c = prob_idx[index].first;
                double prob_c = prob_idx[index].second;
                if (c == blank_id) {
                    probs_b_cur[l] += prob_c * (probs_b_prev[l] + probs_nb_prev[l]);
                } else {
                    std::string last_char = l.substr(l.size()-1, 1);
                    std::string new_char = vocabulary[c];
                    std::string l_plus = l + new_char;

                    if( prefix_set_next.find(l_plus) == prefix_set_next.end()) {
                        probs_b_cur[l_plus] = probs_nb_cur[l_plus] = 0.0;
                    }
                    if (last_char == new_char) {
                        probs_nb_cur[l_plus] += prob_c * probs_b_prev[l];
                        probs_nb_cur[l] += prob_c * probs_nb_prev[l];
                    } else if (new_char == " ") {
                        double score = 1.0;
                        if (ext_scorer != NULL && l.size() > 1) {
                            score = ext_scorer->get_score(l.substr(1));
                        }
                        probs_nb_cur[l_plus] += score * prob_c * (
                            probs_b_prev[l] + probs_nb_prev[l]);
                    } else {
                        probs_nb_cur[l_plus] += prob_c * (
                            probs_b_prev[l] + probs_nb_prev[l]);
                    }
                    prefix_set_next[l_plus] = probs_nb_cur[l_plus] + probs_b_cur[l_plus];
                }
            }

            prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l];
        }

        probs_b_prev = probs_b_cur;
        probs_nb_prev = probs_nb_cur;
        std::vector<std::pair<std::string, double> >
                  prefix_vec_next(prefix_set_next.begin(),
                                  prefix_set_next.end());
        std::sort(prefix_vec_next.begin(),
                  prefix_vec_next.end(),
                  pair_comp_second_rev<std::string, double>);
        int k = beam_size<prefix_vec_next.size() ? beam_size:prefix_vec_next.size();
        prefix_set_prev = std::map<std::string, double>
                  (prefix_vec_next.begin(), prefix_vec_next.begin()+k);
    }

    // post processing
    std::vector<std::pair<double, std::string> > beam_result;
    for (std::map<std::string, double>::iterator it = prefix_set_prev.begin();
         it != prefix_set_prev.end(); it++) {
        if (it->second > 0.0 && it->first.size() > 1) {
            double prob = it->second;
            std::string sentence = it->first.substr(1);
            // scoring the last word
            if (ext_scorer != NULL && sentence[sentence.size()-1] != ' ') {
                prob = prob * ext_scorer->get_score(sentence);
            }
            double log_prob = log(prob);
            beam_result.push_back(std::pair<double, std::string>(log_prob, sentence));
        }
    }
    // sort the result and return
    std::sort(beam_result.begin(), beam_result.end(),
              pair_comp_first_rev<double, std::string>);
    return beam_result;
}
