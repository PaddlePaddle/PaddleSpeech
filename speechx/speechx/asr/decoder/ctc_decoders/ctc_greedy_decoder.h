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

#ifndef CTC_GREEDY_DECODER_H
#define CTC_GREEDY_DECODER_H

#include <string>
#include <vector>

/* CTC Greedy (Best Path) Decoder
 *
 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary: A vector of vocabulary.
 * Return:
 *     The decoding result in string
 */
std::string ctc_greedy_decoding(
    const std::vector<std::vector<double>>& probs_seq,
    const std::vector<std::string>& vocabulary,
    size_t blank_id);

#endif  // CTC_GREEDY_DECODER_H
