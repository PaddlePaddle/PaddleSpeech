#include "decoder/ctc_beam_search_decoder.h"

#include "base/basic_types.h"
#include "decoder/ctc_decoders/decoder_utils.h"

namespace ppspeech {

using std::vector;
using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

CTCBeamSearch::CTCBeamSearch(std::shared_ptr<CTCBeamSearchOptions> opts) :
    opts_(opts),
    vocabulary_(nullptr),
    init_ext_scorer_(nullptr), 
    blank_id(-1),
    space_id(-1),
    root(nullptr) {

    LOG(INFO) << "dict path: " << _opts.dict_file;
    vocabulary_ = std::make_shared<vector<string>>();
    if (!basr::ReadDictToVector(_opts.dict_file, *vocabulary_)) {
        LOG(INFO) << "load the dict failed";
    }
    LOG(INFO) << "read the vocabulary success, dict size: " << vocabulary_->size();

    LOG(INFO) << "language model path: " << _opts.lm_path;
    init_ext_scorer_ = std::make_shared<Scorer>(_opts.alpha, 
                                                _opts.beta, 
                                                _opts.lm_path, 
                                                *vocabulary_);
}

void CTCBeamSearch::InitDecoder() {
    
    blank_id = 0;
    auto it = std::find(vocabulary_->begin(), vocabulary_->end(), " ");
    
    space_id = it - vocabulary_->begin();
    // if no space in vocabulary
    if ((size_t)space_id >= vocabulary_->size()) {
        space_id = -2;
    }  

    clear_prefixes();
    
    root = std::make_shared<PathTrie>();
    root->score = root->log_prob_b_prev = 0.0;
    prefixes.push_back(root.get());
    if (init_ext_scorer_ != nullptr && !init_ext_scorer_->is_character_based()) {
        auto fst_dict =
            static_cast<fst::StdVectorFst *>(init_ext_scorer_->dictionary);
        fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
        root->set_dictionary(dict_ptr);
        
        auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
        root->set_matcher(matcher);
    }
}

void CTCBeamSearch::ResetPrefixes() {
  for (size_t i = 0; i < prefixes.size(); i++) {
    if (prefixes[i] != nullptr) {
      delete prefixes[i];
      prefixes[i] = nullptr;
    }
  }
}

int CTCBeamSearch::DecodeLikelihoods(const vector<vector<float>>&probs, 
                                      vector<string>& nbest_words) {
  std::thread::id this_id = std::this_thread::get_id();
  Timer timer;
  vector<vector<double>> double_probs(probs.size(), vector<double>(probs[0].size(), 0));

  int row = probs.size();
  int col = probs[0].size();
  for(int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++){
      double_probs[i][j] = static_cast<double>(probs[i][j]);
    }
  }

  timer.Reset();
  vector<std::pair<double, string>> results = AdvanceDecoding(double_probs);
  LOG(INFO) <<"ctc decoding elapsed time(s) " << static_cast<float>(timer.Elapsed()) / 1000.0f;
  for (const auto& item : results) {
    nbest_words.push_back(item.second);
  }
  return 0;
} 

vector<std::pair<double, string>> CTCBeamSearch::AdvanceDecoding(const vector<vector<double>>& probs_seq) {
  size_t num_time_steps = probs_seq.size();
  size_t beam_size = _opts.beam_size;
  double cutoff_prob = _opts.cutoff_prob;
  size_t cutoff_top_n = _opts.cutoff_top_n;
    
  for (size_t time_step = 0; time_step < num_time_steps; time_step++) {
    const auto& prob = probs_seq[time_step];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (init_ext_scorer_ != nullptr) {
      size_t num_prefixes = std::min(prefixes.size(), beam_size);
      std::sort(prefixes.begin(), prefixes.begin() + num_prefixes,
                prefix_compare);
            
      if (num_prefixes == 0) {
        continue;
      }
      min_cutoff = prefixes[num_prefixes - 1]->score +
                      std::log(prob[blank_id]) -
                      std::max(0.0, init_ext_scorer_->beta);

      full_beam = (num_prefixes == beam_size);
    }
        
    vector<std::pair<size_t, float>> log_prob_idx =
                get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n);

    // loop over chars
    size_t log_prob_idx_len = log_prob_idx.size();
    for (size_t index = 0; index < log_prob_idx_len; index++) {            
      SearchOneChar(full_beam, log_prob_idx[index], min_cutoff);
        
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
    } // if 
  } // for probs_seq

  // score the last word of each prefix that doesn't end with space
  LMRescore();
  CalculateApproxScore();
  return get_beam_search_result(prefixes, *vocabulary_, beam_size);
}

int CTCBeamSearch::SearchOneChar(const bool& full_beam,
                                 const std::pair<size_t, float>& log_prob_idx,
                                 const float& min_cutoff) {
  size_t beam_size = _opts.beam_size;
  const auto& c = log_prob_idx.first;
  const auto& log_prob_c = log_prob_idx.second;
  size_t prefixes_len = std::min(prefixes.size(), beam_size);
    
  for (size_t i = 0; i < prefixes_len; ++i) {
    auto prefix = prefixes[i];
    if (full_beam && log_prob_c + prefix->score < min_cutoff) {
      break;
    }

    if (c == blank_id) {
      prefix->log_prob_b_cur = log_sum_exp(
                              prefix->log_prob_b_cur, 
                              log_prob_c +
                              prefix->score);
      continue;
    }

    // repeated character
    if (c == prefix->character) {
      // p_{nb}(l;x_{1:t}) = p(c;x_{t})p(l;x_{1:t-1})
      prefix->log_prob_nb_cur = log_sum_exp(
                                  prefix->log_prob_nb_cur,
                                  log_prob_c + 
                                  prefix->log_prob_nb_prev);
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
          (c == space_id || init_ext_scorer_->is_character_based())) {
        PathTrie *prefix_to_score = nullptr;
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
                      log_sum_exp(prefix_new->log_prob_nb_cur, 
                                  log_p);
    }
  } // end of loop over prefix
  return 0;
}

void CTCBeamSearch::CalculateApproxScore() {
  size_t beam_size = _opts.beam_size;
  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(
      prefixes.begin(), 
      prefixes.begin() + num_prefixes, 
      prefix_compare);

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
    double approx_ctc = prefixes[i]->score;
    if (init_ext_scorer_ != nullptr) {
      vector<int> output;
      prefixes[i]->get_path_vec(output);
      auto prefix_length = output.size();
      auto words = init_ext_scorer_->split_labels(output);
      // remove word insert
      approx_ctc = approx_ctc - prefix_length * init_ext_scorer_->beta;
      // remove language model weight:
      approx_ctc -=
          (init_ext_scorer_->get_sent_log_prob(words)) * init_ext_scorer_->alpha;
    }
    prefixes[i]->approx_ctc = approx_ctc;
  }
}

void CTCBeamSearch::LMRescore() {
  size_t beam_size = _opts.beam_size;
  if (init_ext_scorer_ != nullptr && !init_ext_scorer_->is_character_based()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      auto prefix = prefixes[i];
      if (!prefix->is_empty() && prefix->character != space_id) {
        float score = 0.0;
        vector<string> ngram = init_ext_scorer_->make_ngram(prefix);
        score = init_ext_scorer_->get_log_cond_prob(ngram) * init_ext_scorer_->alpha;
        score += init_ext_scorer_->beta;
        prefix->score += score;
      }
    }
  }
}

} // namespace ppspeech