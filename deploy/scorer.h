#ifndef SCORER_H_
#define SCORER_H_

#include <string>


class Scorer{
private:
    float _alpha;
    float _beta;
    void *_language_model;

public:
    Scorer(){}
    Scorer(float alpha, float beta, std::string lm_model_path);
    ~Scorer();
    int word_count(std::string);
    double language_model_score(std::string);
    double get_score(std::string);
};

#endif
