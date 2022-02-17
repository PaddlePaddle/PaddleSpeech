#include "base/basic_types.h"

#pragma once

namespace ppspeech {

struct CTCBeamSearchOptions {
    std::string dict_file;
    std::string lm_path;
    BaseFloat alpha;
    BaseFloat beta;
    BaseFloat cutoff_prob;
    int beam_size;
    int cutoff_top_n;
    int num_proc_bsearch;
    CTCBeamSearchOptions() :
        dict_file("./model/words.txt"),
        lm_path("./model/lm.arpa"),
        alpha(1.9f),
        beta(5.0),
        beam_size(300),
        cutoff_prob(0.99f),
        cutoff_top_n(40),
        num_proc_bsearch(0) {
    }

    void Register(kaldi::OptionsItf* opts) {
        opts->Register("dict", &dict_file, "dict file ");
        opts->Register("lm-path", &lm_path, "language model file");
        opts->Register("alpha", &alpha, "alpha");
        opts->Register("beta", &beta, "beta");
        opts->Register("beam-size", &beam_size, "beam size for beam search method");
        opts->Register("cutoff-prob", &cutoff_prob, "cutoff probs");
        opts->Register("cutoff-top-n", &cutoff_top_n, "cutoff top n");
        opts->Register("num-proc-bsearch", &num_proc_bsearch, "num proc bsearch");
    }
};

class CTCBeamSearch {
public: 

    CTCBeamSearch(std::shared_ptr<CTCBeamSearchOptions> opts);

    ~CTCBeamSearch() {
    }
    bool InitDecoder();
    int DecodeLikelihoods(const std::vector<std::vector<BaseFloat>>&probs, 
                          std::vector<std::string>& nbest_words);

    std::vector<DecodeResult>& GetDecodeResult() {
        return decoder_results_;
    }

private:
  void ResetPrefixes();
  int32 SearchOneChar(const bool& full_beam,
                      const std::pair<size_t, BaseFloat>& log_prob_idx,
                      const BaseFloat& min_cutoff);
  void CalculateApproxScore();
  void LMRescore();
  std::vector<std::pair<double, std::string>> 
    AdvanceDecoding(const std::vector<std::vector<double>>& probs_seq);
  CTCBeamSearchOptions opts_;
  std::shared_ptr<Scorer> init_ext_scorer_; // todo separate later
  std::vector<DecodeResult> decoder_results_;
  std::vector<std::vector<std::string>> vocabulary_; // todo remove later

  size_t blank_id;        
  int space_id;
  std::shared_ptr<PathTrie> root;
  std::vector<PathTrie*> prefixes;
};

} // namespace basr