#ifndef SCORER_H_
#define SCORER_H_

#include <string>

/* External scorer to evaluate a prefix or a complete sentence
 * when a new word appended during decoding, consisting of word
 * count and language model scoring.

 * Example:
 *     LmScorer ext_scorer(alpha, beta, "path_to_language_model.klm");
 *     double score = ext_scorer.get_score("sentence_to_score");
 */
class LmScorer{
private:
    float _alpha;
    float _beta;
    void *_language_model;

    // word insertion term
    int word_count(std::string);
    // n-gram language model scoring
    double language_model_score(std::string);

public:
    LmScorer(){}
    LmScorer(float alpha, float beta, std::string lm_model_path);
    ~LmScorer();

    // reset params alpha & beta
    void reset_params(float alpha, float beta);
    // get the final score
    double get_score(std::string, bool log=false);
};

#endif //SCORER_H_
