// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef PADDLE_TTS_SERVING_FRONT_FRONT_INTERFACE_H
#define PADDLE_TTS_SERVING_FRONT_FRONT_INTERFACE_H

#include <glog/logging.h>
#include <fstream>
#include <map>
#include <memory>
#include <string>
//#include "utils/dir_utils.h"
#include <cppjieba/Jieba.hpp>
#include "absl/strings/str_split.h"
#include "front/text_normalize.h"


namespace ppspeech {

class FrontEngineInterface : public TextNormalizer {
  public:
    explicit FrontEngineInterface(std::string conf) : _conf_file(conf) {
        TextNormalizer();
        _jieba = nullptr;
        _initialed = false;
        init();
    }

    int init();
    ~FrontEngineInterface() {}

    // 读取配置文件
    int ReadConfFile();

    // 简体转繁体
    int Trand2Simp(const std::wstring &sentence, std::wstring *sentence_simp);

    // 生成字典
    int GenDict(const std::string &file,
                std::map<std::string, std::string> &map);

    // 由 词+词性的分词结果转为仅包含词的结果
    int GetSegResult(std::vector<std::pair<std::string, std::string>> *seg,
                     std::vector<std::string> *seg_words);

    // 生成句子的音素，音调id。如果音素和音调未分开，则 toneids
    // 为空（fastspeech2），反之则不为空(speedyspeech)
    int GetSentenceIds(const std::string &sentence,
                       std::vector<int> *phoneids,
                       std::vector<int> *toneids);

    // 根据分词结果获取词的音素，音调id，并对读音进行适当修改
    // (ModifyTone)。如果音素和音调未分开，则 toneids
    // 为空（fastspeech2），反之则不为空(speedyspeech)
    int GetWordsIds(
        const std::vector<std::pair<std::string, std::string>> &cut_result,
        std::vector<int> *phoneids,
        std::vector<int> *toneids);

    // 结巴分词生成包含词和词性的分词结果，再对分词结果进行适当修改
    // (MergeforModify)
    int Cut(const std::string &sentence,
            std::vector<std::pair<std::string, std::string>> *cut_result);

    // 字词到音素的映射，查找字典
    int GetPhone(const std::string &word, std::string &phone);

    // 音素到音素id
    int Phone2Phoneid(const std::string &phone,
                      std::vector<int> *phoneid,
                      std::vector<int> *toneids);


    // 根据韵母判断该词中每个字的读音都为第三声。true表示词中每个字都是第三声
    bool AllToneThree(const std::vector<std::string> &finals);

    // 判断词是否是叠词
    bool IsReduplication(const std::string &word);

    // 获取每个字词的声母韵母列表
    int GetInitialsFinals(const std::string &word,
                          std::vector<std::string> *word_initials,
                          std::vector<std::string> *word_finals);

    // 获取每个字词的韵母列表
    int GetFinals(const std::string &word,
                  std::vector<std::string> *word_finals);

    // 整个词转成向量形式，向量的每个元素对应词的一个字
    int Word2WordVec(const std::string &word,
                     std::vector<std::wstring> &wordvec);

    // 将整个词重新进行 full cut，分词后，各个词会在词典中
    int SplitWord(const std::string &word,
                  std::vector<std::string> *fullcut_word);

    // 对分词结果进行处理：对包含“不”字的分词结果进行整理
    std::vector<std::pair<std::string, std::string>> MergeBu(
        std::vector<std::pair<std::string, std::string>> *seg_result);

    // 对分词结果进行处理：对包含“一”字的分词结果进行整理
    std::vector<std::pair<std::string, std::string>> Mergeyi(
        std::vector<std::pair<std::string, std::string>> *seg_result);

    // 对分词结果进行处理：对前后相同的两个字进行合并
    std::vector<std::pair<std::string, std::string>> MergeReduplication(
        std::vector<std::pair<std::string, std::string>> *seg_result);

    // 对一个词和后一个词他们的读音均为第三声的两个词进行合并
    std::vector<std::pair<std::string, std::string>> MergeThreeTones(
        std::vector<std::pair<std::string, std::string>> *seg_result);

    // 对一个词的最后一个读音和后一个词的第一个读音为第三声的两个词进行合并
    std::vector<std::pair<std::string, std::string>> MergeThreeTones2(
        std::vector<std::pair<std::string, std::string>> *seg_result);

    // 对分词结果进行处理：对包含“儿”字的分词结果进行整理
    std::vector<std::pair<std::string, std::string>> MergeEr(
        std::vector<std::pair<std::string, std::string>> *seg_result);

    // 对分词结果进行处理、修改
    int MergeforModify(
        std::vector<std::pair<std::string, std::string>> *seg_result,
        std::vector<std::pair<std::string, std::string>> *merge_seg_result);


    // 对包含“不”字的相关词音调进行修改
    int BuSandi(const std::string &word, std::vector<std::string> &finals);

    // 对包含“一”字的相关词音调进行修改
    int YiSandhi(const std::string &word, std::vector<std::string> &finals);

    // 对一些特殊词（包括量词，语助词等）的相关词音调进行修改
    int NeuralSandhi(const std::string &word,
                     const std::string &pos,
                     std::vector<std::string> &finals);

    // 对包含第三声的相关词音调进行修改
    int ThreeSandhi(const std::string &word, std::vector<std::string> &finals);

    // 对字词音调进行处理、修改
    int ModifyTone(const std::string &word,
                   const std::string &pos,
                   std::vector<std::string> &finals);


    // 对儿化音进行处理
    std::vector<std::vector<std::string>> MergeErhua(
        const std::vector<std::string> &initials,
        const std::vector<std::string> &finals,
        const std::string &word,
        const std::string &pos);


  private:
    bool _initialed;
    cppjieba::Jieba *_jieba;
    std::vector<std::string> _punc;
    std::vector<std::string> _punc_omit;

    std::string _conf_file;
    std::map<std::string, std::string> conf_map;
    std::map<std::string, std::string> word_phone_map;
    std::map<std::string, std::string> phone_id_map;
    std::map<std::string, std::string> tone_id_map;
    std::map<std::string, std::string> trand_simp_map;


    std::string _jieba_dict_path;
    std::string _jieba_hmm_path;
    std::string _jieba_user_dict_path;
    std::string _jieba_idf_path;
    std::string _jieba_stop_word_path;

    std::string _seperate_tone;
    std::string _word2phone_path;
    std::string _phone2id_path;
    std::string _tone2id_path;
    std::string _trand2simp_path;

    std::vector<std::string> must_erhua;
    std::vector<std::string> not_erhua;

    std::vector<std::string> must_not_neural_tone_words;
    std::vector<std::string> must_neural_tone_words;
};
}  // namespace ppspeech
#endif