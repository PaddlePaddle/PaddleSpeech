#include <iostream>
#include <unistd.h>
#include "scorer.h"
#include "decoder_utils.h"

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
    Config config;
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
    State state, tmp_state, out_state;
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
