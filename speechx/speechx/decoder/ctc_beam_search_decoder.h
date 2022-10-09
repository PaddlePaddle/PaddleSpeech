// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
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

#include "base/common.h"
#include "decoder/ctc_decoders/path_trie.h"
#include "decoder/ctc_decoders/scorer.h"
#include "kaldi/decoder/decodable-itf.h"
#include "util/parse-options.h"

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
    CTCBeamSearchOptions()
        : dict_file("vocab.txt"),
          lm_path(""),
          alpha(1.9f),
          beta(5.0),
          beam_size(300),
          cutoff_prob(0.99f),
          cutoff_top_n(40),
          num_proc_bsearch(10) {}

    void Register(kaldi::OptionsItf* opts) {
        opts->Register("dict", &dict_file, "dict file ");
        opts->Register("lm-path", &lm_path, "language model file");
        opts->Register("alpha", &alpha, "alpha");
        opts->Register("beta", &beta, "beta");
        opts->Register(
            "beam-size", &beam_size, "beam size for beam search method");
        opts->Register("cutoff-prob", &cutoff_prob, "cutoff probs");
        opts->Register("cutoff-top-n", &cutoff_top_n, "cutoff top n");
        opts->Register(
            "num-proc-bsearch", &num_proc_bsearch, "num proc bsearch");
    }
};

class CTCBeamSearch {
  public:
    explicit CTCBeamSearch(const CTCBeamSearchOptions& opts);
    ~CTCBeamSearch() {}
    void InitDecoder();
    void Decode(std::shared_ptr<kaldi::DecodableInterface> decodable);
    std::string GetBestPath();
    std::vector<std::pair<double, std::string>> GetNBestPath();
    std::string GetFinalBestPath();
    int NumFrameDecoded();
    int DecodeLikelihoods(const std::vector<std::vector<BaseFloat>>& probs,
                          std::vector<std::string>& nbest_words);
    void AdvanceDecode(
        const std::shared_ptr<kaldi::DecodableInterface>& decodable);
    void Reset();

  private:
    void ResetPrefixes();
    int32 SearchOneChar(const bool& full_beam,
                        const std::pair<size_t, BaseFloat>& log_prob_idx,
                        const BaseFloat& min_cutoff);
    void CalculateApproxScore();
    void LMRescore();
    void AdvanceDecoding(const std::vector<std::vector<BaseFloat>>& probs);

    CTCBeamSearchOptions opts_;
    std::shared_ptr<Scorer> init_ext_scorer_;  // todo separate later
    std::vector<std::string> vocabulary_;      // todo remove later
    size_t blank_id_;
    int space_id_;
    std::shared_ptr<PathTrie> root_;
    std::vector<PathTrie*> prefixes_;
    int num_frame_decoded_;
    DISALLOW_COPY_AND_ASSIGN(CTCBeamSearch);
};

}  // namespace ppspeech