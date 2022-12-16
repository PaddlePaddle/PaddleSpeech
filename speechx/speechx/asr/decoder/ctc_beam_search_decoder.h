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

// used by deepspeech2

#pragma once

#include "decoder/ctc_beam_search_opt.h"
#include "decoder/ctc_decoders/path_trie.h"
#include "decoder/ctc_decoders/scorer.h"
#include "decoder/decoder_itf.h"

namespace ppspeech {

class CTCBeamSearch : public DecoderBase {
  public:
    explicit CTCBeamSearch(const CTCBeamSearchOptions& opts);
    ~CTCBeamSearch() {}

    void InitDecoder();

    void Reset();

    void AdvanceDecode(
        const std::shared_ptr<kaldi::DecodableInterface>& decodable);

    void Decode(std::shared_ptr<kaldi::DecodableInterface> decodable);

    std::string GetBestPath();
    std::vector<std::pair<double, std::string>> GetNBestPath();
    std::vector<std::pair<double, std::string>> GetNBestPath(int n);
    std::string GetFinalBestPath();

    std::string GetPartialResult() {
        CHECK(false) << "Not implement.";
        return {};
    }

    int DecodeLikelihoods(const std::vector<std::vector<BaseFloat>>& probs,
                          const std::vector<std::string>& nbest_words);

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
    int space_id_;
    std::shared_ptr<PathTrie> root_;
    std::vector<PathTrie*> prefixes_;

    DISALLOW_COPY_AND_ASSIGN(CTCBeamSearch);
};

}  // namespace ppspeech