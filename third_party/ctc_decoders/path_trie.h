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

#ifndef PATH_TRIE_H
#define PATH_TRIE_H

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "fst/fstlib.h"

/* Trie tree for prefix storing and manipulating, with a dictionary in
 * finite-state transducer for spelling correction.
 */
class PathTrie {
  public:
    PathTrie();
    ~PathTrie();

    // get new prefix after appending new char
    PathTrie* get_path_trie(int new_char, bool reset = true);

    // get the prefix in index from root to current node
    PathTrie* get_path_vec(std::vector<int>& output);

    // get the prefix in index from some stop node to current nodel
    PathTrie* get_path_vec(
        std::vector<int>& output,
        int stop,
        size_t max_steps = std::numeric_limits<size_t>::max());

    // update log probs
    void iterate_to_vec(std::vector<PathTrie*>& output);

    // set dictionary for FST
    void set_dictionary(fst::StdVectorFst* dictionary);

    void set_matcher(std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>>);

    bool is_empty() { return ROOT_ == character; }

    // remove current path from root
    void remove();

    float log_prob_b_prev;
    float log_prob_nb_prev;
    float log_prob_b_cur;
    float log_prob_nb_cur;
    float score;
    float approx_ctc;
    int character;
    PathTrie* parent;

  private:
    int ROOT_;
    bool exists_;
    bool has_dictionary_;

    std::vector<std::pair<int, PathTrie*>> children_;

    // pointer to dictionary of FST
    fst::StdVectorFst* dictionary_;
    fst::StdVectorFst::StateId dictionary_state_;
    // true if finding ars in FST
    std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>> matcher_;
};

#endif  // PATH_TRIE_H
