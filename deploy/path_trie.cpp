#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "path_trie.h"
#include "decoder_utils.h"

PathTrie::PathTrie() {
    _log_prob_b_prev = -NUM_FLT_INF;
    _log_prob_nb_prev = -NUM_FLT_INF;
    _log_prob_b_cur = -NUM_FLT_INF;
    _log_prob_nb_cur = -NUM_FLT_INF;
    _score = -NUM_FLT_INF;

    _ROOT = -1;
    _character = _ROOT;
    _exists = true;
    _parent = nullptr;
    _dictionary = nullptr;
    _dictionary_state = 0;
    _has_dictionary = false;
    _matcher = nullptr;  // finds arcs in FST
}

PathTrie::~PathTrie() {
    for (auto child : _children) {
        delete child.second;
    }
}

PathTrie* PathTrie::get_path_trie(int new_char, bool reset) {
    auto child = _children.begin();
    for (child = _children.begin(); child != _children.end(); ++child) {
        if (child->first == new_char) {
            break;
        }
    }
    if ( child != _children.end() ) {
        if (!child->second->_exists) {
            child->second->_exists = true;
            child->second->_log_prob_b_prev = -NUM_FLT_INF;
            child->second->_log_prob_nb_prev = -NUM_FLT_INF;
            child->second->_log_prob_b_cur = -NUM_FLT_INF;
            child->second->_log_prob_nb_cur = -NUM_FLT_INF;
        }
        return (child->second);
    } else {
        if (_has_dictionary) {
            _matcher->SetState(_dictionary_state);
            bool found = _matcher->Find(new_char);
            if (!found) {
                // Adding this character causes word outside dictionary
                auto FSTZERO = fst::TropicalWeight::Zero();
                auto final_weight = _dictionary->Final(_dictionary_state);
                bool is_final = (final_weight != FSTZERO);
                if (is_final && reset) {
                    _dictionary_state = _dictionary->Start();
                }
                return nullptr;
            } else {
                PathTrie* new_path = new PathTrie;
                new_path->_character = new_char;
                new_path->_parent = this;
                new_path->_dictionary = _dictionary;
                new_path->_dictionary_state = _matcher->Value().nextstate;
                new_path->_has_dictionary = true;
                new_path->_matcher = _matcher;
                _children.push_back(std::make_pair(new_char, new_path));
                return new_path;
            }
        } else {
            PathTrie* new_path = new PathTrie;
            new_path->_character = new_char;
            new_path->_parent = this;
            _children.push_back(std::make_pair(new_char, new_path));
            return new_path;
        }
    }
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output) {
    return get_path_vec(output, _ROOT);
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output,
        int stop,
        size_t max_steps) {
    if (_character == stop ||
            _character == _ROOT ||
            output.size() == max_steps) {
        std::reverse(output.begin(), output.end());
        return this;
    } else {
        output.push_back(_character);
        return _parent->get_path_vec(output, stop, max_steps);
    }
}

void PathTrie::iterate_to_vec(
        std::vector<PathTrie*>& output) {
    if (_exists) {
        _log_prob_b_prev = _log_prob_b_cur;
        _log_prob_nb_prev = _log_prob_nb_cur;

        _log_prob_b_cur = -NUM_FLT_INF;
        _log_prob_nb_cur = -NUM_FLT_INF;

        _score = log_sum_exp(_log_prob_b_prev, _log_prob_nb_prev);
        output.push_back(this);
    }
    for (auto child : _children) {
        child.second->iterate_to_vec(output);
    }
}

void PathTrie::remove() {
    _exists = false;

    if (_children.size() == 0) {
        auto child = _parent->_children.begin();
        for (child = _parent->_children.begin();
                child != _parent->_children.end(); ++child) {
            if (child->first == _character) {
                _parent->_children.erase(child);
                break;
            }
        }

        if ( _parent->_children.size() == 0 && !_parent->_exists ) {
            _parent->remove();
        }

        delete this;
    }
}

void PathTrie::set_dictionary(fst::StdVectorFst* dictionary) {
    _dictionary = dictionary;
    _dictionary_state = dictionary->Start();
    _has_dictionary = true;
}

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;
void PathTrie::set_matcher(std::shared_ptr<FSTMATCH> matcher) {
    _matcher = matcher;
}
