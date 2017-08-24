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

    PathTrie* get_path_vec(std::vector<int> &output);

    PathTrie* get_path_vec(std::vector<int>& output,
                        int stop,
                        size_t max_steps = std::numeric_limits<size_t>::max());

    void iterate_to_vec(std::vector<PathTrie*> &output);

    void set_dictionary(fst::StdVectorFst* dictionary);

    void set_matcher(std::shared_ptr<FSTMATCH> matcher);

    bool is_empty() {
        return _ROOT == _character;
    }

    void remove();

    float _log_prob_b_prev;
    float _log_prob_nb_prev;
    float _log_prob_b_cur;
    float _log_prob_nb_cur;
    float _score;
    float _approx_ctc;


    int  _ROOT;
    int  _character;
    bool _exists;

    PathTrie *_parent;
    std::vector<std::pair<int, PathTrie*> > _children;

    fst::StdVectorFst* _dictionary;
    fst::StdVectorFst::StateId _dictionary_state;
    bool _has_dictionary;
    std::shared_ptr<FSTMATCH> _matcher;
};

#endif // PATH_TRIE_H
