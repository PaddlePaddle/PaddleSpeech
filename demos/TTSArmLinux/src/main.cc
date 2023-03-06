#include <cstdlib>
#include <iostream>
#include <memory>
#include "paddle_api.h"
#include "Predictor.hpp"

using namespace paddle::lite_api;

std::vector<std::vector<float>> sentencesToChoose = {
    // 009901 昨日，这名“伤者”与医生全部被警方依法刑事拘留。
    {261, 231, 175, 116, 179, 262, 44, 154, 126, 177, 19, 262, 42, 241, 72, 177, 56, 174, 245, 37, 186, 37, 49, 151, 127, 69, 19, 179, 72, 69, 4, 260, 126, 177, 116, 151, 239, 153, 141},
    // 009902 钱伟长想到上海来办学校是经过深思熟虑的。
    {174, 83, 213, 39, 20, 260, 89, 40, 30, 177, 22, 71, 9, 153, 8, 37, 17, 260, 251, 260, 99, 179, 177, 116, 151, 125, 70, 233, 177, 51, 176, 108, 177, 184, 153, 242, 40, 45},
    // 009903 她见我一进门就骂，吃饭时也骂，骂得我抬不起头。
    {182, 2, 151, 85, 232, 73, 151, 123, 154, 52, 151, 143, 154, 5, 179, 39, 113, 69, 17, 177, 114, 105, 154, 5, 179, 154, 5, 40, 45, 232, 182, 8, 37, 186, 174, 74, 182, 168},
    // 009904 李述德在离开之前，只说了一句“柱驼杀父亲了”。
    {153, 74, 177, 186, 40, 42, 261, 10, 153, 73, 152, 7, 262, 113, 174, 83, 179, 262, 115, 177, 230, 153, 45, 73, 151, 242, 180, 262, 186, 182, 231, 177, 2, 69, 186, 174, 124, 153, 45},
    // 009905 这种车票和保险单捆绑出售属于重复性购买。
    {262, 44, 262, 163, 39, 41, 173, 99, 71, 42, 37, 28, 260, 84, 40, 14, 179, 152, 220, 37, 21, 39, 183, 177, 170, 179, 177, 185, 240, 39, 162, 69, 186, 260, 128, 70, 170, 154, 9},
    // 009906 戴佩妮的男友西米露接唱情歌，让她非常开心。
    {40, 10, 173, 49, 155, 72, 40, 45, 155, 15, 142, 260, 72, 154, 74, 153, 186, 179, 151, 103, 39, 22, 174, 126, 70, 41, 179, 175, 22, 182, 2, 69, 46, 39, 20, 152, 7, 260, 120},
    // 009907 观大势、谋大局、出大策始终是该院的办院方针。
    {70, 199, 40, 5, 177, 116, 154, 168, 40, 5, 151, 240, 179, 39, 183, 40, 5, 38, 44, 179, 177, 115, 262, 161, 177, 116, 70, 7, 247, 40, 45, 37, 17, 247, 69, 19, 262, 51},
    // 009908 他们骑着摩托回家，正好为农忙时的父母帮忙。
    {182, 2, 154, 55, 174, 73, 262, 45, 154, 157, 182, 230, 71, 212, 151, 77, 180, 262, 59, 71, 29, 214, 155, 162, 154, 20, 177, 114, 40, 45, 69, 186, 154, 185, 37, 19, 154, 20},
    // 009909 但是因为还没到退休年龄，只能掰着指头捱日子。
    {40, 17, 177, 116, 120, 214, 71, 8, 154, 47, 40, 30, 182, 214, 260, 140, 155, 83, 153, 126, 180, 262, 115, 155, 57, 37, 7, 262, 45, 262, 115, 182, 171, 8, 175, 116, 261, 112},
    // 009910 这几天雨水不断，人们恨不得待在家里不出门。
    {262, 44, 151, 74, 182, 82, 240, 177, 213, 37, 184, 40, 202, 180, 175, 52, 154, 55, 71, 54, 37, 186, 40, 42, 40, 7, 261, 10, 151, 77, 153, 74, 37, 186, 39, 183, 154, 52},
};

void usage(const char *binName) {
    std::cerr << "Usage:" << std::endl
        << "\t" << binName << " <AM-model-path> <VOC-model-path> <sentences-index:1-10> <output-wav-path>" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        usage(argv[0]);
        return -1;
    }
    const char *AMModelPath = argv[1];
    const char *VOCModelPath = argv[2];
    int sentencesIndex = atoi(argv[3]) - 1;
    const char *outputWavPath = argv[4];

    if (sentencesIndex < 0 || sentencesIndex >= sentencesToChoose.size()) {
        std::cerr << "sentences-index out of range" << std::endl;
        return -1;
    }

    Predictor predictor;
    if (!predictor.Init(AMModelPath, VOCModelPath, 1, "LITE_POWER_HIGH")) {
        std::cerr << "predictor init failed" << std::endl;
        return -1;
    }
    
    if (!predictor.RunModel(sentencesToChoose[sentencesIndex])) {
        std::cerr << "predictor run model failed" << std::endl;
        return -1;
    }

    std::cout << "Inference time: " << predictor.GetInferenceTime() << " ms, "
              << "WAV size (without header): " << predictor.GetWavSize() << " bytes" << std::endl;

    if (!predictor.WriteWavToFile(outputWavPath)) {
        std::cerr << "write wav file failed" << std::endl;
        return -1;
    }

    return 0;
}
