// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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

// modified from
// https://github.com/wenet-e2e/wenet/blob/main/runtime/core/decoder/ctc_prefix_beam_search.h

#pragma once

#include "base/common.h"
#include "utils/math.h"

namespace ppspeech {

class ContextGraph;

struct PrefixScore {
    // decoding, unit in log scale
    float b = -kBaseFloatMax;   // blank ending score
    float nb = -kBaseFloatMax;  // none-blank ending score

    // decoding score, sum
    float Score() const { return LogSumExp(b, nb); }

    // timestamp, unit in log sclae
    float v_b = -kBaseFloatMax;             // viterbi blank ending score
    float v_nb = -kBaseFloatMax;            // niterbi none-blank ending score
    float cur_token_prob = -kBaseFloatMax;  // prob of current token
    std::vector<int> times_b;               // times of viterbi blank path
    std::vector<int> times_nb;              // times of viterbi non-blank path


    // timestamp score, max
    float ViterbiScore() const { return std::max(v_b, v_nb); }

    // get timestamp
    const std::vector<int>& Times() const {
        return v_b > v_nb ? times_b : times_nb;
    }

    // context state
    bool has_context = false;
    int context_state = 0;
    float context_score = 0;
    std::vector<int> start_boundaries;
    std::vector<int> end_boundaries;


    // decodign score with context bias
    float TotalScore() const { return Score() + context_score; }

    void CopyContext(const PrefixScore& prefix_score) {
        context_state = prefix_score.context_state;
        context_score = prefix_score.context_score;
        start_boundaries = prefix_score.start_boundaries;
        end_boundaries = prefix_score.end_boundaries;
    }

    void UpdateContext(const std::shared_ptr<ContextGraph>& constext_graph,
                       const PrefixScore& prefix_score,
                       int word_id,
                       int prefix_len) {
        CHECK(false);
    }

    void InitEmpty() {
        b = 0.0f;             // log(1)
        nb = -kBaseFloatMax;  // log(0)
        v_b = 0.0f;           // log(1)
        v_nb = 0.0f;          // log(1)
    }
};

struct PrefixScoreHash {
    // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
    std::size_t operator()(const std::vector<int>& prefix) const {
        std::size_t seed = prefix.size();
        for (auto& i : prefix) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using PrefixWithScoreType = std::pair<std::vector<int>, PrefixScoreHash>;

}  // namespace ppspeech
