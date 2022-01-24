// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "COPYING.APACHE2.0");
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

#include "ctc_greedy_decoder.h"
#include "decoder_utils.h"

std::string ctc_greedy_decoding(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t blank_id) {
    // dimension check
    size_t num_time_steps = probs_seq.size();
    for (size_t i = 0; i < num_time_steps; ++i) {
        VALID_CHECK_EQ(probs_seq[i].size(),
                       vocabulary.size(),
                       "The shape of probs_seq does not match with "
                       "the shape of the vocabulary");
    }

    // size_t blank_id = vocabulary.size();

    std::vector<size_t> max_idx_vec(num_time_steps, 0);
    std::vector<size_t> idx_vec;
    for (size_t i = 0; i < num_time_steps; ++i) {
        double max_prob = 0.0;
        size_t max_idx = 0;
        const std::vector<double> &probs_step = probs_seq[i];
        for (size_t j = 0; j < probs_step.size(); ++j) {
            if (max_prob < probs_step[j]) {
                max_idx = j;
                max_prob = probs_step[j];
            }
        }
        // id with maximum probability in current time step
        max_idx_vec[i] = max_idx;
        // deduplicate
        if ((i == 0) || ((i > 0) && max_idx_vec[i] != max_idx_vec[i - 1])) {
            idx_vec.push_back(max_idx_vec[i]);
        }
    }

    std::string best_path_result;
    for (size_t i = 0; i < idx_vec.size(); ++i) {
        if (idx_vec[i] != blank_id) {
            std::string ch = vocabulary[idx_vec[i]];
            best_path_result += (ch == kSPACE) ? tSPACE : ch;
        }
    }
    return best_path_result;
}
