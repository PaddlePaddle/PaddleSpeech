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

#pragma once

#include "decoder/ctc_beam_search_opt.h"
#include "decoder/ctc_prefix_beam_search_score.h"
#include "decoder/decoder_itf.h"

#include "kaldi/decoder/decodable-itf.h"

namespace ppspeech {

class CTCPrefixBeamSearch : public DecoderInterface {
  public:
    explicit CTCPrefixBeamSearch(const CTCBeamSearchOptions& opts);
    ~CTCPrefixBeamSearch() {}

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
    size_t blank_id_;
    int num_frame_decoded_;
    DISALLOW_COPY_AND_ASSIGN(CTCPrefixBeamSearch);
};

}  // namespace basr