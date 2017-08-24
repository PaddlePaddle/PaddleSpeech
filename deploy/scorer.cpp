#include <iostream>
#include <unistd.h>
#include "lm/config.hh"
#include "lm/state.hh"
#include "lm/model.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "scorer.h"
#include "decoder_utils.h"

using namespace lm::ngram;

Scorer::Scorer(double alpha, double beta, const std::string& lm_path) {
    this->alpha = alpha;
    this->beta = beta;
    _is_character_based = true;
    _language_model = nullptr;
    _max_order = 0;
    // load language model
    load_LM(lm_path.c_str());
}

Scorer::~Scorer() {
    if (_language_model != nullptr)
        delete static_cast<lm::base::Model*>(_language_model);
}

void Scorer::load_LM(const char* filename) {
    if (access(filename, F_OK) != 0) {
        std::cerr << "Invalid language model file !!!" << std::endl;
        exit(1);
    }
    RetriveStrEnumerateVocab enumerate;
    lm::ngram::Config config;
    config.enumerate_vocab = &enumerate;
    _language_model = lm::ngram::LoadVirtual(filename, config);
    _max_order = static_cast<lm::base::Model*>(_language_model)->Order();
    _vocabulary = enumerate.vocabulary;
    for (size_t i = 0; i < _vocabulary.size(); ++i) {
        if (_is_character_based
            && _vocabulary[i] != UNK_TOKEN
            && _vocabulary[i] != START_TOKEN
            && _vocabulary[i] != END_TOKEN
            && get_utf8_str_len(enumerate.vocabulary[i]) > 1) {
                _is_character_based = false;
        }
    }
}

double Scorer::get_log_cond_prob(const std::vector<std::string>& words) {
    lm::base::Model* model = static_cast<lm::base::Model*>(_language_model);
    double cond_prob;
    lm::ngram::State state, tmp_state, out_state;
    // avoid to inserting <s> in begin
    model->NullContextWrite(&state);
    for (size_t i = 0; i < words.size(); ++i) {
        lm::WordIndex word_index = model->BaseVocabulary().Index(words[i]);
        // encounter OOV
        if (word_index == 0) {
            return OOV_SCOER;
        }
        cond_prob = model->BaseScore(&state, word_index, &out_state);
        tmp_state = state;
        state = out_state;
        out_state = tmp_state;
    }
    // log10 prob
    return cond_prob;
}

double Scorer::get_sent_log_prob(const std::vector<std::string>& words) {
    std::vector<std::string> sentence;
    if (words.size() == 0) {
        for (size_t i = 0; i < _max_order; ++i) {
            sentence.push_back(START_TOKEN);
        }
    } else {
        for (size_t i = 0; i < _max_order - 1; ++i) {
            sentence.push_back(START_TOKEN);
        }
        sentence.insert(sentence.end(), words.begin(), words.end());
    }
    sentence.push_back(END_TOKEN);
    return get_log_prob(sentence);
}

double Scorer::get_log_prob(const std::vector<std::string>& words) {
    assert(words.size() > _max_order);
    double score = 0.0;
    for (size_t i = 0; i < words.size() - _max_order + 1; ++i) {
        std::vector<std::string> ngram(words.begin() + i,
                                       words.begin() + i + _max_order);
        score += get_log_cond_prob(ngram);
    }
    return score;
}

/* Strip a input sentence
 * Parameters:
 *     str: A reference to the objective string
 *     ch: The character to prune
 * Return:
 *     void
 */
inline void strip(std::string &str, char ch=' ') {
    if (str.size() == 0) return;
    int start  = 0;
    int end = str.size()-1;
    for (int i=0; i<str.size(); i++){
        if (str[i] == ch) {
            start ++;
        } else {
            break;
        }
    }
    for (int i=str.size()-1; i>=0; i--) {
        if  (str[i] == ch) {
            end --;
        } else {
            break;
        }
    }

    if (start == 0 && end == str.size()-1) return;
    if (start > end) {
        std::string emp_str;
        str = emp_str;
    } else {
        str = str.substr(start, end-start+1);
    }
}

int Scorer::word_count(std::string sentence) {
    strip(sentence);
    int cnt = 1;
    for (int i=0; i<sentence.size(); i++) {
        if (sentence[i] == ' ' && sentence[i-1] != ' ') {
            cnt ++;
        }
    }
    return cnt;
}

double Scorer::get_log_cond_prob(std::string sentence) {
    lm::base::Model *model = (lm::base::Model *)this->_language_model;
    State state, out_state;
    lm::FullScoreReturn ret;
    model->BeginSentenceWrite(&state);

    for (util::TokenIter<util::SingleCharacter, true> it(sentence, ' '); it; ++it){
        lm::WordIndex wid = model->BaseVocabulary().Index(*it);
        ret = model->BaseFullScore(&state, wid, &out_state);
        state = out_state;
    }
    //log10 prob
    double log_prob = ret.prob;
    return log_prob;
}

void Scorer::reset_params(float alpha, float beta) {
    this->alpha = alpha;
    this->beta = beta;
}

double Scorer::get_score(std::string sentence, bool log) {
    double lm_score = get_log_cond_prob(sentence);
    int word_cnt = word_count(sentence);

    double final_score = 0.0;
    if (log == false) {
        final_score = pow(10, alpha * lm_score) * pow(word_cnt, beta);
    } else {
        final_score = alpha * lm_score * std::log(10)
                      + beta * std::log(word_cnt);
    }
    return final_score;
}

//--------------------------------------------------
// Turn indices back into strings of chars
//--------------------------------------------------
std::vector<std::string> Scorer::make_ngram(PathTrie* prefix) {
    /*
    std::vector<std::string> ngram;
    PathTrie* current_node = prefix;
    PathTrie* new_node = nullptr;

    for (int order = 0; order < _max_order; order++) {
        std::vector<int> prefix_vec;

        if (_is_character_based) {
            new_node = current_node->get_path_vec(prefix_vec, ' ', 1);
            current_node = new_node;
        } else {
            new_node = current_node->getPathVec(prefix_vec, ' ');
            current_node = new_node->_parent;  // Skipping spaces
        }

        // reconstruct word
        std::string word = vec2str(prefix_vec);
        ngram.push_back(word);

        if (new_node->_character == -1) {
            // No more spaces, but still need order
            for (int i = 0; i < max_order - order - 1; i++) {
                ngram.push_back("<s>");
            }
            break;
        }
    }
    std::reverse(ngram.begin(), ngram.end());
    */
    std::vector<std::string> ngram;
    ngram.push_back("this");
    return ngram;
}  //---------------- End makeNgrams ------------------
