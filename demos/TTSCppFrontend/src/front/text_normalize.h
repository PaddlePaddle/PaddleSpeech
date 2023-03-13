#ifndef PADDLE_TTS_SERVING_FRONT_TEXT_NORMALIZE_H
#define PADDLE_TTS_SERVING_FRONT_TEXT_NORMALIZE_H

#include <map>
#include <regex>
#include <string>
#include <codecvt>
#include <glog/logging.h>
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "base/type_conv.h"

namespace ppspeech {

class TextNormalizer {
public:
    TextNormalizer() {
        InitMap();
    }
    ~TextNormalizer() {

    }

    int InitMap();
    int Replace(std::wstring &sentence, const int &pos, const int &len, const std::wstring &repstr);
    int SplitByPunc(const std::wstring &sentence, std::vector<std::wstring> &sentence_part);

    std::string CreateTextValue(const std::string &num,  bool use_zero=true);
    std::string SingleDigit2Text(const std::string &num_str, bool alt_one = false);
    std::string SingleDigit2Text(const std::wstring &num, bool alt_one = false);
    std::string MultiDigit2Text(const std::string &num_str, bool alt_one = false, bool use_zero = true);
    std::string MultiDigit2Text(const std::wstring &num, bool alt_one = false, bool use_zero = true);
    std::string Digits2Text(const std::string &num_str);
    std::string Digits2Text(const std::wstring &num);

    int ReData(std::wstring &sentence);
    int ReData2(std::wstring &sentence);
    int ReTime(std::wstring &sentence);
    int ReTemperature(std::wstring &sentence);
    int ReFrac(std::wstring &sentence);
    int RePercentage(std::wstring &sentence);
    int ReMobilePhone(std::wstring &sentence);
    int RePhone(std::wstring &sentence);
    int ReRange(std::wstring &sentence);
    int ReInterger(std::wstring &sentence);
    int ReDecimalNum(std::wstring &sentence);
    int RePositiveQuantifiers(std::wstring &sentence);
    int ReDefalutNum(std::wstring &sentence);
    int ReNumber(std::wstring &sentence);
    int SentenceNormalize(std::wstring &sentence);


private:
    std::map<std::string, std::string> digits_map;
    std::map<int, std::string> units_map;


};

}

#endif