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

#pragma once

#include "base/common.h"
#include "utils/math.h"

namespace ppspeech {

struct PrefxiScore {
    // decoding, unit in log scale
    float b = -kFloatMax;   // blank ending score
    float nb = -kFloatMax;  // none-blank ending score

    // timestamp, unit in log sclae
    float v_b = -kFloatMax;             // viterbi blank ending score
    float v_nb = -kFloatMax;            // niterbi none-blank ending score
    float cur_token_prob = -kFloatMax;  // prob of current token
    std::vector<int> times_b;           // times of viterbi blank path
    std::vector<int> times_nb;          // times of viterbi non-blank path

    // context state
    bool has_context = false;
    int context_state = 0;
    float context_score = 0;

    // decoding score, sum
    float Score() const { return LogSumExp(b, nb); }

    // decodign score with context bias
    float TotalScore() const { return Score() + context_score; }

    // timestamp score, max
    float ViterbiScore() const { return std::max(v_b, v_nb); }

    // get timestamp
    const std::vector<int>& Times() const {
        return v_b > v_nb ? times_b : times_nb;
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
