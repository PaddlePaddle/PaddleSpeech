#include <iostream>
#include <unistd.h>
#include "scorer.h"
#include "lm/model.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"

using namespace lm::ngram;

Scorer::Scorer(float alpha, float beta, std::string lm_model_path) {
    this->_alpha = alpha;
    this->_beta = beta;

    if (access(lm_model_path.c_str(), F_OK) != 0) {
        std::cout<<"Invalid language model path!"<<std::endl;
        exit(1);
    }
    this->_language_model = LoadVirtual(lm_model_path.c_str());
}

Scorer::~Scorer(){
   delete (lm::base::Model *)this->_language_model;
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

double Scorer::language_model_score(std::string sentence) {
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
    this->_alpha = alpha;
    this->_beta = beta;
}

double Scorer::get_score(std::string sentence) {
    double lm_score = language_model_score(sentence);
    int word_cnt = word_count(sentence);

    double final_score = pow(10, _alpha*lm_score) * pow(word_cnt, _beta);
    return final_score;
}
