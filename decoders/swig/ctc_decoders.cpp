#include "ctc_decoders.h"

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

std::string ctc_greedy_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary) {
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; i++) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size() + 1,
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  size_t blank_id = vocabulary.size();

  std::vector<size_t> max_idx_vec;
  for (size_t i = 0; i < num_time_steps; i++) {
    double max_prob = 0.0;
    size_t max_idx = 0;
    for (size_t j = 0; j < probs_seq[i].size(); j++) {
      if (max_prob < probs_seq[i][j]) {
        max_idx = j;
        max_prob = probs_seq[i][j];
      }
    }
    max_idx_vec.push_back(max_idx);
  }

  std::vector<size_t> idx_vec;
  for (size_t i = 0; i < max_idx_vec.size(); i++) {
    if ((i == 0) || ((i > 0) && max_idx_vec[i] != max_idx_vec[i - 1])) {
      idx_vec.push_back(max_idx_vec[i]);
    }
  }

  std::string best_path_result;
  for (size_t i = 0; i < idx_vec.size(); i++) {
    if (idx_vec[i] != blank_id) {
      best_path_result += vocabulary[idx_vec[i]];
    }
  }
  return best_path_result;
}

std::vector<std::pair<double, std::string>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const size_t beam_size,
    std::vector<std::string> vocabulary,
    const double cutoff_prob,
    const size_t cutoff_top_n,
    Scorer *ext_scorer) {
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; i++) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size() + 1,
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  // assign blank id
  size_t blank_id = vocabulary.size();

  // assign space id
  std::vector<std::string>::iterator it =
      std::find(vocabulary.begin(), vocabulary.end(), " ");
  int space_id = it - vocabulary.begin();
  // if no space in vocabulary
  if (space_id >= vocabulary.size()) {
    space_id = -2;
  }

  // init prefixes' root
  PathTrie root;
  root.score = root.log_prob_b_prev = 0.0;
  std::vector<PathTrie *> prefixes;
  prefixes.push_back(&root);

  if (ext_scorer != nullptr) {
    if (ext_scorer->is_char_map_empty()) {
      ext_scorer->set_char_map(vocabulary);
    }
    if (!ext_scorer->is_character_based()) {
      if (ext_scorer->dictionary == nullptr) {
        // fill dictionary for fst with space
        ext_scorer->fill_dictionary(true);
      }
      auto fst_dict = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
      fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
      root.set_dictionary(dict_ptr);
      auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
      root.set_matcher(matcher);
    }
  }

  // prefix search over time
  for (size_t time_step = 0; time_step < num_time_steps; time_step++) {
    std::vector<double> prob = probs_seq[time_step];
    std::vector<std::pair<int, double>> prob_idx;
    for (size_t i = 0; i < prob.size(); i++) {
      prob_idx.push_back(std::pair<int, double>(i, prob[i]));
    }

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer != nullptr) {
      size_t num_prefixes = std::min(prefixes.size(), beam_size);
      std::sort(
          prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
      min_cutoff = prefixes[num_prefixes - 1]->score + log(prob[blank_id]) -
                   std::max(0.0, ext_scorer->beta);
      full_beam = (num_prefixes == beam_size);
    }

    // pruning of vacobulary
    size_t cutoff_len = prob.size();
    if (cutoff_prob < 1.0 || cutoff_top_n < prob.size()) {
      std::sort(
          prob_idx.begin(), prob_idx.end(), pair_comp_second_rev<int, double>);
      if (cutoff_prob < 1.0) {
        double cum_prob = 0.0;
        cutoff_len = 0;
        for (size_t i = 0; i < prob_idx.size(); i++) {
          cum_prob += prob_idx[i].second;
          cutoff_len += 1;
          if (cum_prob >= cutoff_prob) break;
        }
      }
      cutoff_len = std::min(cutoff_len, cutoff_top_n);
      prob_idx = std::vector<std::pair<int, double>>(
          prob_idx.begin(), prob_idx.begin() + cutoff_len);
    }
    std::vector<std::pair<size_t, float>> log_prob_idx;
    for (size_t i = 0; i < cutoff_len; i++) {
      log_prob_idx.push_back(std::pair<int, float>(
          prob_idx[i].first, log(prob_idx[i].second + NUM_FLT_MIN)));
    }

    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      float log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes.size() && i < beam_size; i++) {
        auto prefix = prefixes[i];

        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
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
            PathTrie *prefix_toscore = nullptr;

            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_toscore = prefix_new;
            } else {
              prefix_toscore = prefix;
            }

            double score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_toscore);
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;

            log_p += score;
            log_p += ext_scorer->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over chars

    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       prefix_compare);

      for (size_t i = beam_size; i < prefixes.size(); i++) {
        prefixes[i]->remove();
      }
    }
  }  // end of loop over time

  // compute aproximate ctc score as the return score
  for (size_t i = 0; i < beam_size && i < prefixes.size(); i++) {
    double approx_ctc = prefixes[i]->score;

    if (ext_scorer != nullptr) {
      std::vector<int> output;
      prefixes[i]->get_path_vec(output);
      size_t prefix_length = output.size();
      auto words = ext_scorer->split_labels(output);
      // remove word insert
      approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
      // remove language model weight:
      approx_ctc -= (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
    }

    prefixes[i]->approx_ctc = approx_ctc;
  }

  // allow for the post processing
  std::vector<PathTrie *> space_prefixes;
  if (space_prefixes.empty()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); i++) {
      space_prefixes.push_back(prefixes[i]);
    }
  }

  std::sort(space_prefixes.begin(), space_prefixes.end(), prefix_compare);
  std::vector<std::pair<double, std::string>> output_vecs;
  for (size_t i = 0; i < beam_size && i < space_prefixes.size(); i++) {
    std::vector<int> output;
    space_prefixes[i]->get_path_vec(output);
    // convert index to string
    std::string output_str;
    for (size_t j = 0; j < output.size(); j++) {
      output_str += vocabulary[output[j]];
    }
    std::pair<double, std::string> output_pair(-space_prefixes[i]->approx_ctc,
                                               output_str);
    output_vecs.emplace_back(output_pair);
  }

  return output_vecs;
}

std::vector<std::vector<std::pair<double, std::string>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const size_t beam_size,
    const std::vector<std::string> &vocabulary,
    const size_t num_processes,
    const double cutoff_prob,
    const size_t cutoff_top_n,
    Scorer *ext_scorer) {
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();

  // scorer filling up
  if (ext_scorer != nullptr) {
    if (ext_scorer->is_char_map_empty()) {
      ext_scorer->set_char_map(vocabulary);
    }
    if (!ext_scorer->is_character_based() &&
        ext_scorer->dictionary == nullptr) {
      // init dictionary
      ext_scorer->fill_dictionary(true);
    }
  }

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, std::string>>>> res;
  for (size_t i = 0; i < batch_size; i++) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  probs_split[i],
                                  beam_size,
                                  vocabulary,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<std::pair<double, std::string>>> batch_results;
  for (size_t i = 0; i < batch_size; i++) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
