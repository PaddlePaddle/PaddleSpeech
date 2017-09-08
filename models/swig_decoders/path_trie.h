#ifndef PATH_TRIE_H
#define PATH_TRIE_H
#pragma once
#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <fst/fstlib.h>

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

class PathTrie {
public:
  PathTrie();
  ~PathTrie();

  PathTrie* get_path_trie(int new_char, bool reset = true);

  PathTrie* get_path_vec(std::vector<int>& output);

  PathTrie* get_path_vec(std::vector<int>& output,
                         int stop,
                         size_t max_steps = std::numeric_limits<size_t>::max());

  void iterate_to_vec(std::vector<PathTrie*>& output);

  void set_dictionary(fst::StdVectorFst* dictionary);

  void set_matcher(std::shared_ptr<FSTMATCH> matcher);

  bool is_empty() { return _ROOT == character; }

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
  int _ROOT;
  bool _exists;
  bool _has_dictionary;

  std::vector<std::pair<int, PathTrie*>> _children;

  fst::StdVectorFst* _dictionary;
  fst::StdVectorFst::StateId _dictionary_state;
  std::shared_ptr<FSTMATCH> _matcher;
};

#endif  // PATH_TRIE_H
