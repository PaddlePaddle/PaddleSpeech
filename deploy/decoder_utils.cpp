#include <limits>
#include <algorithm>
#include <cmath>
#include "decoder_utils.h"

size_t get_utf8_str_len(const std::string& str) {
    size_t str_len = 0;
    for (char c : str) {
        str_len += ((c & 0xc0) != 0x80);
    }
    return str_len;
}

//-------------------------------------------------------
//  Overriding less than operator for sorting
//-------------------------------------------------------
bool prefix_compare(const PathTrie* x,  const PathTrie* y) {
    if (x->_score == y->_score) {
        if (x->_character == y->_character) {
            return false;
        } else {
            return (x->_character < y->_character);
        }
    } else {
        return x->_score > y->_score;
    }
}  //---------- End path_compare ---------------------------

// --------------------------------------------------------------
// Adds word to fst without copying entire dictionary
// --------------------------------------------------------------
void add_word_to_fst(const std::vector<int>& word,
                     fst::StdVectorFst* dictionary) {
    if (dictionary->NumStates() == 0) {
        fst::StdVectorFst::StateId start = dictionary->AddState();
        assert(start == 0);
        dictionary->SetStart(start);
    }
    fst::StdVectorFst::StateId src = dictionary->Start();
    fst::StdVectorFst::StateId dst;
    for (auto c : word) {
        dst = dictionary->AddState();
        dictionary->AddArc(src, fst::StdArc(c, c, 0, dst));
        src = dst;
    }
    dictionary->SetFinal(dst, fst::StdArc::Weight::One());
}  // ------------ End of add_word_to_fst -----------------------

// ---------------------------------------------------------
// Adds a word to the dictionary FST based on char_map
// ---------------------------------------------------------
bool addWordToDictionary(const std::string& word,
                         const std::unordered_map<std::string, int>& char_map,
                         bool add_space,
                         int SPACE,
                         fst::StdVectorFst* dictionary) {
    /*
    auto characters = UTF8_split(word);

    std::vector<int> int_word;

    for (auto& c : characters) {
        if (c == " ") {
            int_word.push_back(SPACE);
        } else {
            auto int_c = char_map.find(c);
            if (int_c != char_map.end()) {
                int_word.push_back(int_c->second);
            } else {
                return false;  // return without adding
            }
        }
    }

    if (add_space) {
        int_word.push_back(SPACE);
    }

    add_word_to_fst(int_word, dictionary);
    */
    return true;
}  // -------------- End of addWordToDictionary ------------
