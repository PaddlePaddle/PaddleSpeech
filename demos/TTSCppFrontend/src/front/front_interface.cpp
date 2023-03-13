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
#include "front/front_interface.h"

namespace ppspeech {

int FrontEngineInterface::init() {
    if (_initialed) {
        return 0;
    }
    if (0 != ReadConfFile()) {
        LOG(ERROR) << "Read front conf file failed";
        return -1;
    }

    _jieba = new cppjieba::Jieba(_jieba_dict_path,
                                 _jieba_hmm_path,
                                 _jieba_user_dict_path,
                                 _jieba_idf_path,
                                 _jieba_stop_word_path);

    _punc = {"，",
             "。",
             "、",
             "？",
             "：",
             "；",
             "~",
             "！",
             ",",
             ".",
             "?",
             "!",
             ":",
             ";",
             "/",
             "\\"};
    _punc_omit = {"“", "”", "\"", "\""};

    // 需要儿化音处理的词语
    must_erhua = {
        "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿"};
    not_erhua = {"虐儿",   "为儿",   "护儿",   "瞒儿",   "救儿",   "替儿",
                 "有儿",   "一儿",   "我儿",   "俺儿",   "妻儿",   "拐儿",
                 "聋儿",   "乞儿",   "患儿",   "幼儿",   "孤儿",   "婴儿",
                 "婴幼儿", "连体儿", "脑瘫儿", "流浪儿", "体弱儿", "混血儿",
                 "蜜雪儿", "舫儿",   "祖儿",   "美儿",   "应采儿", "可儿",
                 "侄儿",   "孙儿",   "侄孙儿", "女儿",   "男儿",   "红孩儿",
                 "花儿",   "虫儿",   "马儿",   "鸟儿",   "猪儿",   "猫儿",
                 "狗儿"};

    must_not_neural_tone_words = {
        "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子"};
    // 需要轻声处理的词语
    must_neural_tone_words = {
        "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头",
        "馄饨", "风筝", "难为", "队伍", "阔气", "闺女", "门道", "锄头", "铺盖",
        "铃铛", "铁匠", "钥匙", "里脊", "里头", "部分", "那么", "道士", "造化",
        "迷糊", "连累", "这么", "这个", "运气", "过去", "软和", "转悠", "踏实",
        "跳蚤", "跟头", "趔趄", "财主", "豆腐", "讲究", "记性", "记号", "认识",
        "规矩", "见识", "裁缝", "补丁", "衣裳", "衣服", "衙门", "街坊", "行李",
        "行当", "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄", "萝卜", "荸荠", "苗条",
        "苗头", "苍蝇", "芝麻", "舒服", "舒坦", "舌头", "自在", "膏药", "脾气",
        "脑袋", "脊梁", "能耐", "胳膊", "胭脂", "胡萝", "胡琴", "胡同", "聪明",
        "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆", "老头", "老太",
        "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊", "糊涂",
        "精神", "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语",
        "笑话", "窟窿", "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气",
        "秀才", "福气", "祖宗", "砚台", "码头", "石榴", "石头", "石匠", "知识",
        "眼睛", "眯缝", "眨巴", "眉毛", "相声", "盘算", "白净", "痢疾", "痛快",
        "疟疾", "疙瘩", "疏忽", "畜生", "生意", "甘蔗", "琵琶", "琢磨", "琉璃",
        "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜", "牌楼",
        "爽快", "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心", "炊帚", "灯笼",
        "火候", "漂亮", "滑溜", "溜达", "温和", "清楚", "消息", "浪头", "活泼",
        "比方", "正经", "欺负", "模糊", "槟榔", "棺材", "棒槌", "棉花", "核桃",
        "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事", "木头", "木匠",
        "朋友", "月饼", "月亮", "暖和", "明白", "时候", "新鲜", "故事", "收拾",
        "收成", "提防", "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄",
        "招牌", "招呼", "抬举", "护士", "折腾", "扫帚", "打量", "打算", "打点",
        "打扮", "打听", "打发", "扎实", "扁担", "戒指", "懒得", "意识", "意思",
        "情形", "悟性", "怪物", "思量", "怎么", "念头", "念叨", "快活", "忙活",
        "志气", "心思", "得罪", "张罗", "弟兄", "开通", "应酬", "庄稼", "干事",
        "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌", "差事", "工夫",
        "岁数", "屁股", "尾巴", "少爷", "小气", "小伙", "将就", "对头", "对付",
        "寡妇", "家伙", "客气", "实在", "官司", "学问", "学生", "字号", "嫁妆",
        "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫", "妯娌", "妥当",
        "妖精", "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫",
        "多少", "多么", "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴",
        "嘱咐", "嘟囔", "嘀咕", "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴",
        "哈欠", "哆嗦", "咳嗽", "和尚", "告诉", "告示", "含糊", "吓唬", "后头",
        "名字", "名堂", "合同", "吆喝", "叫唤", "口袋", "厚道", "厉害", "千斤",
        "包袱", "包涵", "匀称", "勤快", "动静", "动弹", "功夫", "力气", "前头",
        "刺猬", "刺激", "别扭", "利落", "利索", "利害", "分析", "出息", "凑合",
        "凉快", "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟", "便宜",
        "使唤", "佩服", "作坊", "体面", "位置", "似的", "伙计", "休息", "什么",
        "人家", "亲戚", "亲家", "交情", "云彩", "事情", "买卖", "主意", "丫头",
        "丧气", "两口", "东西", "东家", "世故", "不由", "不在", "下水", "下巴",
        "上头", "上司", "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲",
        "咕噜", "邋遢", "费用", "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅",
        "幸福", "熟悉", "计划", "扑腾", "蜡烛", "姥爷", "照顾", "喉咙", "吉他",
        "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜", "糟蹋", "倒腾", "报复", "逻辑",
        "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记"};


    // 生成词典（词到音素的映射）
    if (0 != GenDict(_word2phone_path, word_phone_map)) {
        LOG(ERROR) << "Genarate word2phone dict failed";
        return -1;
    }

    // 生成音素字典（音素到音素id的映射）
    if (0 != GenDict(_phone2id_path, phone_id_map)) {
        LOG(ERROR) << "Genarate phone2id dict failed";
        return -1;
    }

    // 生成音调字典（音调到音调id的映射）
    if (_seperate_tone == "true") {
        if (0 != GenDict(_tone2id_path, tone_id_map)) {
            LOG(ERROR) << "Genarate tone2id dict failed";
            return -1;
        }
    }

    // 生成繁简字典（繁体到简体id的映射）
    if (0 != GenDict(_trand2simp_path, trand_simp_map)) {
        LOG(ERROR) << "Genarate trand2simp dict failed";
        return -1;
    }

    _initialed = true;
    return 0;
}

int FrontEngineInterface::ReadConfFile() {
    std::ifstream is(_conf_file.c_str(), std::ifstream::in);
    if (!is.good()) {
        LOG(ERROR) << "Cannot open config file: " << _conf_file;
        return -1;
    }
    std::string line, key, value;
    while (std::getline(is, line)) {
        if (line.substr(0, 2) == "--") {
            size_t pos = line.find_first_of("=", 0);
            std::string key = line.substr(2, pos - 2);
            std::string value = line.substr(pos + 1);
            conf_map[key] = value;
            LOG(INFO) << "Key: " << key << "; Value: " << value;
        }
    }

    // jieba conf path
    _jieba_dict_path = conf_map["jieba_dict_path"];
    _jieba_hmm_path = conf_map["jieba_hmm_path"];
    _jieba_user_dict_path = conf_map["jieba_user_dict_path"];
    _jieba_idf_path = conf_map["jieba_idf_path"];
    _jieba_stop_word_path = conf_map["jieba_stop_word_path"];

    // dict path
    _seperate_tone = conf_map["seperate_tone"];
    _word2phone_path = conf_map["word2phone_path"];
    _phone2id_path = conf_map["phone2id_path"];
    _tone2id_path = conf_map["tone2id_path"];
    _trand2simp_path = conf_map["trand2simpd_path"];

    return 0;
}

int FrontEngineInterface::Trand2Simp(const std::wstring &sentence,
                                     std::wstring &sentence_simp) {
    // sentence_simp = sentence;
    for (int i = 0; i < sentence.length(); i++) {
        std::wstring temp(1, sentence[i]);
        std::string sigle_word = ppspeech::wstring2utf8string(temp);
        // 单个字是否在繁转简的字典里
        if (trand_simp_map.find(sigle_word) == trand_simp_map.end()) {
            sentence_simp += temp;
        } else {
            sentence_simp +=
                (ppspeech::utf8string2wstring(trand_simp_map[sigle_word]));
        }
    }

    return 0;
}

int FrontEngineInterface::GenDict(const std::string &dict_file,
                                  std::map<std::string, std::string> &map) {
    std::ifstream is(dict_file.c_str(), std::ifstream::in);
    if (!is.good()) {
        LOG(ERROR) << "Cannot open dict file: " << dict_file;
        return -1;
    }
    std::string line, key, value;
    while (std::getline(is, line)) {
        size_t pos = line.find_first_of(" ", 0);
        key = line.substr(0, pos);
        value = line.substr(pos + 1);
        map[key] = value;
    }
    return 0;
}

int FrontEngineInterface::GetSegResult(
    std::vector<std::pair<std::string, std::string>> &seg,
    std::vector<std::string> &seg_words) {
    std::vector<std::pair<std::string, std::string>>::iterator iter;
    for (iter = seg.begin(); iter != seg.end(); iter++) {
        seg_words.push_back((*iter).first);
    }
    return 0;
}

int FrontEngineInterface::GetSentenceIds(const std::string &sentence,
                                         std::vector<int> &phoneids,
                                         std::vector<int> &toneids) {
    std::vector<std::pair<std::string, std::string>>
        cut_result;  //分词结果包含词和词性
    if (0 != Cut(sentence, cut_result)) {
        LOG(ERROR) << "Cut sentence: \"" << sentence << "\" failed";
        return -1;
    }

    if (0 != GetWordsIds(cut_result, phoneids, toneids)) {
        LOG(ERROR) << "Get words phoneids failed";
        return -1;
    }
    return 0;
}

int FrontEngineInterface::GetWordsIds(
    const std::vector<std::pair<std::string, std::string>> &cut_result,
    std::vector<int> &phoneids,
    std::vector<int> &toneids) {
    std::string word;
    std::string pos;
    std::vector<std::string> word_initials;
    std::vector<std::string> word_finals;
    std::string phone;
    for (int i = 0; i < cut_result.size(); i++) {
        word = cut_result[i].first;
        pos = cut_result[i].second;
        if (std::find(_punc_omit.begin(), _punc_omit.end(), word) ==
            _punc_omit.end()) {  // 非可忽略的标点
            word_initials = {};
            word_finals = {};
            phone = "";
            // 判断是否在标点符号集合中
            if (std::find(_punc.begin(), _punc.end(), word) ==
                _punc.end()) {  // 文字
                // 获取字词的声母韵母列表
                if (0 != GetInitialsFinals(word, word_initials, word_finals)) {
                    LOG(ERROR)
                        << "Genarate the word_initials and word_finals of "
                        << word << " failed";
                    return -1;
                }

                // 对读音进行修改
                if (0 != ModifyTone(word, pos, word_finals)) {
                    LOG(ERROR) << "Failed to modify tone.";
                }

                // 对儿化音进行修改
                std::vector<std::vector<std::string>> new_initals_finals =
                    MergeErhua(word_initials, word_finals, word, pos);
                word_initials = new_initals_finals[0];
                word_finals = new_initals_finals[1];

                // 将声母和韵母合并成音素
                assert(word_initials.size() == word_finals.size());
                std::string temp_phone;
                for (int j = 0; j < word_initials.size(); j++) {
                    if (word_initials[j] != "") {
                        temp_phone = word_initials[j] + " " + word_finals[j];
                    } else {
                        temp_phone = word_finals[j];
                    }
                    if (j == 0) {
                        phone += temp_phone;
                    } else {
                        phone += (" " + temp_phone);
                    }
                }
            } else {  // 标点符号
                if (_seperate_tone == "true") {
                    phone = "sp0";  // speedyspeech
                } else {
                    phone = "sp";  // fastspeech2
                }
            }

            // 音素到音素id
            if (0 != Phone2Phoneid(phone, phoneids, toneids)) {
                LOG(ERROR) << "Genarate the phone id of " << word << " failed";
                return -1;
            }
        }
    }

    return 0;
}

int FrontEngineInterface::Cut(
    const std::string &sentence,
    std::vector<std::pair<std::string, std::string>> &cut_result) {
    std::vector<std::pair<std::string, std::string>> cut_result_jieba;

    // 结巴分词
    _jieba->Tag(sentence, cut_result_jieba);

    // 对分词后结果进行整合
    if (0 != MergeforModify(cut_result_jieba, cut_result)) {
        LOG(ERROR) << "Failed to modify  for word segmentation result.";
        return -1;
    }

    return 0;
}

int FrontEngineInterface::GetPhone(const std::string &word,
                                   std::string &phone) {
    // 判断 word 在不在 词典里，如果不在，进行CutAll分词
    if (word_phone_map.find(word) == word_phone_map.end()) {
        std::vector<std::string> wordcut;
        _jieba->CutAll(word, wordcut);
        phone = word_phone_map[wordcut[0]];
        for (int i = 1; i < wordcut.size(); i++) {
            phone += (" " + word_phone_map[wordcut[i]]);
        }
    } else {
        phone = word_phone_map[word];
    }

    return 0;
}

int FrontEngineInterface::Phone2Phoneid(const std::string &phone,
                                        std::vector<int> &phoneid,
                                        std::vector<int> &toneid) {
    std::vector<std::string> phone_vec;
    phone_vec = absl::StrSplit(phone, " ");
    std::string temp_phone;
    for (int i = 0; i < phone_vec.size(); i++) {
        temp_phone = phone_vec[i];
        if (_seperate_tone == "true") {
            phoneid.push_back(atoi(
                (phone_id_map[temp_phone.substr(0, temp_phone.length() - 1)])
                    .c_str()));
            toneid.push_back(
                atoi((tone_id_map[temp_phone.substr(temp_phone.length() - 1,
                                                    temp_phone.length())])
                         .c_str()));
        } else {
            phoneid.push_back(atoi((phone_id_map[temp_phone]).c_str()));
        }
    }
    return 0;
}


// 根据韵母判断该词中每个字的读音都为第三声。true表示词中每个字都是第三声
bool FrontEngineInterface::AllToneThree(
    const std::vector<std::string> &finals) {
    bool flags = true;
    for (int i = 0; i < finals.size(); i++) {
        if (static_cast<int>(finals[i].back()) != 51) {  //如果读音不为第三声
            flags = false;
        }
    }
    return flags;
}

// 判断词是否是叠词
bool FrontEngineInterface::IsReduplication(const std::string &word) {
    bool flags = false;
    std::wstring word_wstr = ppspeech::utf8string2wstring(word);
    int len = word_wstr.length();
    if (len == 2 && word_wstr[0] == word_wstr[1]) {
        flags = true;
    }
    return flags;
}

// 获取每个字词的声母和韵母列表， word_initials 为声母列表，word_finals
// 为韵母列表
int FrontEngineInterface::GetInitialsFinals(
    const std::string &word,
    std::vector<std::string> &word_initials,
    std::vector<std::string> &word_finals) {
    std::string phone;
    GetPhone(word, phone);  //获取字词对应的音素
    std::vector<std::string> phone_vec = absl::StrSplit(phone, " ");
    //获取韵母，每个字的音素有1或者2个，start为单个字音素的起始位置。
    int start = 0;
    while (start < phone_vec.size()) {
        if (phone_vec[start] == "sp" || phone_vec[start] == "sp0") {
            start += 1;
        }
        // 最后一位不是数字或者最后一位的数字是0，均表示声母，第二个是韵母
        else if (isdigit(phone_vec[start].back()) == 0 ||
                 static_cast<int>(phone_vec[start].back()) == 48) {
            word_initials.push_back(phone_vec[start]);
            word_finals.push_back(phone_vec[start + 1]);
            start += 2;
        } else {
            word_initials.push_back("");
            word_finals.push_back(phone_vec[start]);
            start += 1;
        }
    }

    assert(word_finals.size() == ppspeech::utf8string2wstring(word).length() &&
           word_finals.size() == word_initials.size());

    return 0;
}

// 获取每个字词的韵母列表
int FrontEngineInterface::GetFinals(const std::string &word,
                                    std::vector<std::string> &word_finals) {
    std::vector<std::string> word_initials;
    if (0 != GetInitialsFinals(word, word_initials, word_finals)) {
        LOG(ERROR) << "Failed to get word finals";
        return -1;
    }

    return 0;
}

int FrontEngineInterface::Word2WordVec(const std::string &word,
                                       std::vector<std::wstring> &wordvec) {
    std::wstring word_wstr = ppspeech::utf8string2wstring(word);
    for (int i = 0; i < word_wstr.length(); i++) {
        std::wstring word_sigle(1, word_wstr[i]);
        wordvec.push_back(word_sigle);
    }
    return 0;
}

// yuantian01解释：把一个词再进行分词找到。例子：小雨伞 --> 小 雨伞 或者 小雨 伞
int FrontEngineInterface::SplitWord(const std::string &word,
                                    std::vector<std::string> &new_word_vec) {
    std::vector<std::string> word_vec;
    std::string second_subword;
    _jieba->CutForSearch(word, word_vec);
    // 升序
    std::sort(word_vec.begin(),
              word_vec.end(),
              [](std::string a, std::string b) { return a.size() > b.size(); });
    std::string first_subword = word_vec[0];  // 提取长度最短的字符串
    int first_begin_idx = word.find_first_of(first_subword);
    if (first_begin_idx == 0) {
        second_subword = word.substr(first_subword.length());
        new_word_vec.push_back(first_subword);
        new_word_vec.push_back(second_subword);
    } else {
        second_subword = word.substr(0, word.length() - first_subword.length());
        new_word_vec.push_back(second_subword);
        new_word_vec.push_back(first_subword);
    }

    return 0;
}


// example: 不 一起 --> 不一起
std::vector<std::pair<std::string, std::string>> FrontEngineInterface::MergeBu(
    std::vector<std::pair<std::string, std::string>> &seg_result) {
    std::vector<std::pair<std::string, std::string>> result;
    std::string word;
    std::string pos;
    std::string last_word = "";

    for (int i = 0; i < seg_result.size(); i++) {
        word = seg_result[i].first;
        pos = seg_result[i].second;
        if (last_word == "不") {
            word = last_word + word;
        }
        if (word != "不") {
            result.push_back(make_pair(word, pos));
        }
        last_word = word;
    }

    if (last_word == "不") {
        result.push_back(make_pair(last_word, "d"));
        last_word = "";
    }

    return result;
}

std::vector<std::pair<std::string, std::string>> FrontEngineInterface::Mergeyi(
    std::vector<std::pair<std::string, std::string>> &seg_result) {
    std::vector<std::pair<std::string, std::string>> result_temp;
    std::string word;
    std::string pos;

    // function 1  example: 听 一 听 --> 听一听
    for (int i = 0; i < seg_result.size(); i++) {
        word = seg_result[i].first;
        pos = seg_result[i].second;
        if ((i - 1 >= 0) && (word == "一") && (i + 1 < seg_result.size()) &&
            (seg_result[i - 1].first == seg_result[i + 1].first) &&
            seg_result[i - 1].second == "v") {
            result_temp[i - 1].first =
                result_temp[i - 1].first + "一" + result_temp[i - 1].first;
        } else {
            if ((i - 2 >= 0) && (seg_result[i - 1].first == "一") &&
                (seg_result[i - 2].first == word) && (pos == "v")) {
                continue;
            } else {
                result_temp.push_back(make_pair(word, pos));
            }
        }
    }

    // function 2  example: 一 你 -->  一你
    std::vector<std::pair<std::string, std::string>> result = {};
    for (int j = 0; j < result_temp.size(); j++) {
        word = result_temp[j].first;
        pos = result_temp[j].second;
        if ((result.size() != 0) && (result.back().first == "一")) {
            result.back().first = result.back().first + word;
        } else {
            result.push_back(make_pair(word, pos));
        }
    }

    return result;
}

// example: 你 你 --> 你你
std::vector<std::pair<std::string, std::string>>
FrontEngineInterface::MergeReduplication(
    std::vector<std::pair<std::string, std::string>> &seg_result) {
    std::vector<std::pair<std::string, std::string>> result;
    std::string word;
    std::string pos;

    for (int i = 0; i < seg_result.size(); i++) {
        word = seg_result[i].first;
        pos = seg_result[i].second;
        if ((result.size() != 0) && (word == result.back().first)) {
            result.back().first = result.back().first + seg_result[i].first;
        } else {
            result.push_back(make_pair(word, pos));
        }
    }

    return result;
}

// the first and the second words are all_tone_three
std::vector<std::pair<std::string, std::string>>
FrontEngineInterface::MergeThreeTones(
    std::vector<std::pair<std::string, std::string>> &seg_result) {
    std::vector<std::pair<std::string, std::string>> result;
    std::string word;
    std::string pos;
    std::vector<std::vector<std::string>> finals;  //韵母数组
    std::vector<std::string> word_final;
    std::vector<bool> merge_last(seg_result.size(), false);

    // 判断最后一个分词结果是不是标点，不看标点的声母韵母
    int word_num = seg_result.size() - 1;
    if (std::find(_punc.begin(), _punc.end(), seg_result[word_num].first) ==
        _punc.end()) {  // 最后一个分词结果不是标点
        word_num += 1;
    }

    // 获取韵母数组
    for (int i = 0; i < word_num; i++) {
        word_final = {};
        word = seg_result[i].first;
        pos = seg_result[i].second;
        if (std::find(_punc_omit.begin(), _punc_omit.end(), word) ==
            _punc_omit.end()) {  // 非可忽略的标点，即文字
            if (0 != GetFinals(word, word_final)) {
                LOG(ERROR) << "Failed to get the final of word.";
            }
        }

        finals.push_back(word_final);
    }
    assert(word_num == finals.size());

    // 对第三声读音的字词分词结果进行处理
    for (int i = 0; i < word_num; i++) {
        word = seg_result[i].first;
        pos = seg_result[i].second;
        if (i - 1 >= 0 && AllToneThree(finals[i - 1]) &&
            AllToneThree(finals[i]) && !merge_last[i - 1]) {
            // if the last word is reduplication, not merge, because
            // reduplication need to be _neural_sandhi
            if (!IsReduplication(seg_result[i - 1].first) &&
                (ppspeech::utf8string2wstring(seg_result[i - 1].first))
                            .length() +
                        (ppspeech::utf8string2wstring(word)).length() <=
                    3) {
                result.back().first = result.back().first + seg_result[i].first;
                merge_last[i] = true;
            } else {
                result.push_back(make_pair(word, pos));
            }
        } else {
            result.push_back(make_pair(word, pos));
        }
    }

    //把标点的分词结果补上
    if (word_num < seg_result.size()) {
        result.push_back(
            make_pair(seg_result[word_num].first, seg_result[word_num].second));
    }

    return result;
}

// the last char of first word and the first char of second word is tone_three
std::vector<std::pair<std::string, std::string>>
FrontEngineInterface::MergeThreeTones2(
    std::vector<std::pair<std::string, std::string>> &seg_result) {
    std::vector<std::pair<std::string, std::string>> result;
    std::string word;
    std::string pos;
    std::vector<std::vector<std::string>> finals;  //韵母数组
    std::vector<std::string> word_final;
    std::vector<bool> merge_last(seg_result.size(), false);

    // 判断最后一个分词结果是不是标点
    int word_num = seg_result.size() - 1;
    if (std::find(_punc.begin(), _punc.end(), seg_result[word_num].first) ==
        _punc.end()) {  // 最后一个分词结果不是标点
        word_num += 1;
    }

    // 获取韵母数组
    for (int i = 0; i < word_num; i++) {
        word_final = {};
        word = seg_result[i].first;
        pos = seg_result[i].second;
        // 如果是文字，则获取韵母，如果是可忽略的标点，例如引号，则跳过
        if (std::find(_punc_omit.begin(), _punc_omit.end(), word) ==
            _punc_omit.end()) {
            if (0 != GetFinals(word, word_final)) {
                LOG(ERROR) << "Failed to get the final of word.";
            }
        }

        finals.push_back(word_final);
    }
    assert(word_num == finals.size());

    // 对第三声读音的字词分词结果进行处理
    for (int i = 0; i < word_num; i++) {
        word = seg_result[i].first;
        pos = seg_result[i].second;
        if (i - 1 >= 0 && !finals[i - 1].empty() &&
            absl::EndsWith(finals[i - 1].back(), "3") == true &&
            !finals[i].empty() &&
            absl::EndsWith(finals[i].front(), "3") == true &&
            !merge_last[i - 1]) {
            // if the last word is reduplication, not merge, because
            // reduplication need to be _neural_sandhi
            if (!IsReduplication(seg_result[i - 1].first) &&
                (ppspeech::utf8string2wstring(seg_result[i - 1].first))
                            .length() +
                        ppspeech::utf8string2wstring(word).length() <=
                    3) {
                result.back().first = result.back().first + seg_result[i].first;
                merge_last[i] = true;
            } else {
                result.push_back(make_pair(word, pos));
            }
        } else {
            result.push_back(make_pair(word, pos));
        }
    }

    //把标点的分词结果补上
    if (word_num < seg_result.size()) {
        result.push_back(
            make_pair(seg_result[word_num].first, seg_result[word_num].second));
    }

    return result;
}

// example: 吃饭 儿 --> 吃饭儿
std::vector<std::pair<std::string, std::string>> FrontEngineInterface::MergeEr(
    std::vector<std::pair<std::string, std::string>> &seg_result) {
    std::vector<std::pair<std::string, std::string>> result;
    std::string word;
    std::string pos;

    for (int i = 0; i < seg_result.size(); i++) {
        word = seg_result[i].first;
        pos = seg_result[i].second;
        if ((i - 1 >= 0) && (word == "儿")) {
            result.back().first = result.back().first + seg_result[i].first;
        } else {
            result.push_back(make_pair(word, pos));
        }
    }

    return result;
}

int FrontEngineInterface::MergeforModify(
    std::vector<std::pair<std::string, std::string>> &seg_word_type,
    std::vector<std::pair<std::string, std::string>> &modify_seg_word_type) {
    std::vector<std::string> seg_result;
    GetSegResult(seg_word_type, seg_result);
    LOG(INFO) << "Before merge, seg result is: "
              << limonp::Join(seg_result.begin(), seg_result.end(), "/");

    modify_seg_word_type = MergeBu(seg_word_type);
    modify_seg_word_type = Mergeyi(modify_seg_word_type);
    modify_seg_word_type = MergeReduplication(modify_seg_word_type);
    modify_seg_word_type = MergeThreeTones(modify_seg_word_type);
    modify_seg_word_type = MergeThreeTones2(modify_seg_word_type);
    modify_seg_word_type = MergeEr(modify_seg_word_type);

    seg_result = {};
    GetSegResult(modify_seg_word_type, seg_result);
    LOG(INFO) << "After merge, seg result is: "
              << limonp::Join(seg_result.begin(), seg_result.end(), "/");

    return 0;
}


int FrontEngineInterface::BuSandi(const std::string &word,
                                  std::vector<std::string> &finals) {
    std::wstring bu = L"不";
    std::vector<std::wstring> wordvec;
    // 一个词转成向量形式
    if (0 != Word2WordVec(word, wordvec)) {
        LOG(ERROR) << "Failed to get word vector";
        return -1;
    }

    // e.g. 看不懂   b u4  -->  b u5, 将韵母的最后一位替换成 5
    if (wordvec.size() == 3 && wordvec[1] == bu) {
        finals[1] = finals[1].replace(finals[1].length() - 1, 1, "5");
    } else {
        // e.g. 不怕  b u4 --> b u2, 将韵母的最后一位替换成 2
        for (int i = 0; i < wordvec.size(); i++) {
            if (wordvec[i] == bu && i + 1 < wordvec.size() &&
                absl::EndsWith(finals[i + 1], "4") == true) {
                finals[i] = finals[i].replace(finals[i].length() - 1, 1, "2");
            }
        }
    }

    return 0;
}


int FrontEngineInterface::YiSandhi(const std::string &word,
                                   std::vector<std::string> &finals) {
    std::wstring yi = L"一";
    std::vector<std::wstring> wordvec;
    // 一个词转成向量形式
    if (0 != Word2WordVec(word, wordvec)) {
        LOG(ERROR) << "Failed to get word vector";
        return -1;
    }

    //情况1："一" in number sequences, e.g. 一零零, 二一零
    std::wstring num_wstr = L"零一二三四六七八九";
    std::wstring word_wstr = ppspeech::utf8string2wstring(word);
    if (word_wstr.find(yi) != word_wstr.npos && wordvec.back() != yi) {
        int flags = 0;
        for (int j = 0; j < wordvec.size(); j++) {
            if (num_wstr.find(wordvec[j]) == num_wstr.npos) {
                flags = -1;
                break;
            }
        }
        if (flags == 0) {
            return 0;
        }
    } else if (wordvec.size() == 3 && wordvec[1] == yi &&
               wordvec[0] == wordvec[2]) {
        // "一" between reduplication words shold be yi5, e.g. 看一看
        finals[1] = finals[1].replace(finals[1].length() - 1, 1, "5");
    } else if (wordvec[0] == L"第" && wordvec[1] == yi) {  //以第一位开始
        finals[1] = finals[1].replace(finals[1].length() - 1, 1, "1");
    } else {
        for (int i = 0; i < wordvec.size(); i++) {
            if (wordvec[i] == yi && i + 1 < wordvec.size()) {
                if (absl::EndsWith(finals[i + 1], "4") == true) {
                    // "一" before tone4 should be yi2, e.g. 一段
                    finals[i] =
                        finals[i].replace(finals[i].length() - 1, 1, "2");
                } else {
                    // "一" before non-tone4 should be yi4, e.g. 一天
                    finals[i] =
                        finals[i].replace(finals[i].length() - 1, 1, "4");
                }
            }
        }
    }

    return 0;
}

int FrontEngineInterface::NeuralSandhi(const std::string &word,
                                       const std::string &pos,
                                       std::vector<std::string> &finals) {
    std::wstring word_wstr = ppspeech::utf8string2wstring(word);
    std::vector<std::wstring> wordvec;
    // 一个词转成向量形式
    if (0 != Word2WordVec(word, wordvec)) {
        LOG(ERROR) << "Failed to get word vector";
        return -1;
    }
    int word_num = wordvec.size();
    assert(word_num == word_wstr.length());

    // 情况1：reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
    for (int j = 0; j < wordvec.size(); j++) {
        std::string inits = "nva";
        if (j - 1 >= 0 && wordvec[j] == wordvec[j - 1] &&
            inits.find(pos[0]) != inits.npos) {
            finals[j] = finals[j].replace(finals[j].length() - 1, 1, "5");
        }
    }

    // 情况2：对下述词的处理
    std::wstring yuqici = L"吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶";
    std::wstring de = L"的地得";
    std::wstring le = L"了着过";
    std::vector<std::string> le_pos = {"ul", "uz", "ug"};
    std::wstring men = L"们子";
    std::vector<std::string> men_pos = {"r", "n"};
    std::wstring weizhi = L"上下里";
    std::vector<std::string> weizhi_pos = {"s", "l", "f"};
    std::wstring dong = L"来去";
    std::wstring fangxiang = L"上下进出回过起开";
    std::wstring ge = L"个";
    std::wstring xiushi = L"几有两半多各整每做是零一二三四六七八九";
    auto ge_idx = word_wstr.find_first_of(ge);  // 出现“个”的第一个位置

    if (word_num >= 1 && yuqici.find(wordvec.back()) != yuqici.npos) {
        finals.back() =
            finals.back().replace(finals.back().length() - 1, 1, "5");
    } else if (word_num >= 1 && de.find(wordvec.back()) != de.npos) {
        finals.back() =
            finals.back().replace(finals.back().length() - 1, 1, "5");
    } else if (word_num == 1 && le.find(wordvec[0]) != le.npos &&
               find(le_pos.begin(), le_pos.end(), pos) != le_pos.end()) {
        finals.back() =
            finals.back().replace(finals.back().length() - 1, 1, "5");
    } else if (word_num > 1 && men.find(wordvec.back()) != men.npos &&
               find(men_pos.begin(), men_pos.end(), pos) != men_pos.end() &&
               find(must_not_neural_tone_words.begin(),
                    must_not_neural_tone_words.end(),
                    word) != must_not_neural_tone_words.end()) {
        finals.back() =
            finals.back().replace(finals.back().length() - 1, 1, "5");
    } else if (word_num > 1 && weizhi.find(wordvec.back()) != weizhi.npos &&
               find(weizhi_pos.begin(), weizhi_pos.end(), pos) !=
                   weizhi_pos.end()) {
        finals.back() =
            finals.back().replace(finals.back().length() - 1, 1, "5");
    } else if (word_num > 1 && dong.find(wordvec.back()) != dong.npos &&
               fangxiang.find(wordvec[word_num - 2]) != fangxiang.npos) {
        finals.back() =
            finals.back().replace(finals.back().length() - 1, 1, "5");
    }
    // 情况3：对“个”字前面带有修饰词的字词读音处理
    else if ((ge_idx != word_wstr.npos && ge_idx >= 1 &&
              xiushi.find(wordvec[ge_idx - 1]) != xiushi.npos) ||
             word_wstr == ge) {
        finals.back() =
            finals.back().replace(finals.back().length() - 1, 1, "5");
    } else {
        if (find(must_neural_tone_words.begin(),
                 must_neural_tone_words.end(),
                 word) != must_neural_tone_words.end() ||
            (word_num >= 2 &&
             find(must_neural_tone_words.begin(),
                  must_neural_tone_words.end(),
                  ppspeech::wstring2utf8string(word_wstr.substr(
                      word_num - 2))) != must_neural_tone_words.end())) {
            finals.back() =
                finals.back().replace(finals.back().length() - 1, 1, "5");
        }
    }

    // 进行进一步分词，把长词切分更短些
    std::vector<std::string> word_list;
    if (0 != SplitWord(word, word_list)) {
        LOG(ERROR) << "Failed to split word.";
        return -1;
    }
    // 创建对应的 韵母列表
    std::vector<std::vector<std::string>> finals_list;
    std::vector<std::string> finals_temp;
    finals_temp.assign(
        finals.begin(),
        finals.begin() + ppspeech::utf8string2wstring(word_list[0]).length());
    finals_list.push_back(finals_temp);
    finals_temp.assign(
        finals.begin() + ppspeech::utf8string2wstring(word_list[0]).length(),
        finals.end());
    finals_list.push_back(finals_temp);

    finals = {};
    for (int i = 0; i < word_list.size(); i++) {
        std::wstring temp_wstr = ppspeech::utf8string2wstring(word_list[i]);
        if ((find(must_neural_tone_words.begin(),
                  must_neural_tone_words.end(),
                  word_list[i]) != must_neural_tone_words.end()) ||
            (temp_wstr.length() >= 2 &&
             find(must_neural_tone_words.begin(),
                  must_neural_tone_words.end(),
                  ppspeech::wstring2utf8string(
                      temp_wstr.substr(temp_wstr.length() - 2))) !=
                 must_neural_tone_words.end())) {
            finals_list[i].back() = finals_list[i].back().replace(
                finals_list[i].back().length() - 1, 1, "5");
        }
        finals.insert(
            finals.end(), finals_list[i].begin(), finals_list[i].end());
    }

    return 0;
}

int FrontEngineInterface::ThreeSandhi(const std::string &word,
                                      std::vector<std::string> &finals) {
    std::wstring word_wstr = ppspeech::utf8string2wstring(word);
    std::vector<std::vector<std::string>> finals_list;
    std::vector<std::string> finals_temp;
    std::vector<std::wstring> wordvec;
    // 一个词转成向量形式
    if (0 != Word2WordVec(word, wordvec)) {
        LOG(ERROR) << "Failed to get word vector";
        return -1;
    }
    int word_num = wordvec.size();
    assert(word_num == word_wstr.length());

    if (word_num == 2 && AllToneThree(finals)) {
        finals[0] = finals[0].replace(finals[0].length() - 1, 1, "2");
    } else if (word_num == 3) {
        // 进行进一步分词，把长词切分更短些
        std::vector<std::string> word_list;
        if (0 != SplitWord(word, word_list)) {
            LOG(ERROR) << "Failed to split word.";
            return -1;
        }
        if (AllToneThree(finals)) {
            std::wstring temp_wstr = ppspeech::utf8string2wstring(word_list[0]);
            // disyllabic + monosyllabic, e.g. 蒙古/包
            if (temp_wstr.length() == 2) {
                finals[0] = finals[0].replace(finals[0].length() - 1, 1, "2");
                finals[1] = finals[1].replace(finals[1].length() - 1, 1, "2");
            } else if (temp_wstr.length() ==
                       1) {  // monosyllabic + disyllabic, e.g. 纸/老虎
                finals[1] = finals[1].replace(finals[1].length() - 1, 1, "2");
            }
        } else {
            // 创建对应的 韵母列表
            finals_temp = {};
            finals_list = {};
            finals_temp.assign(
                finals.begin(),
                finals.begin() +
                    ppspeech::utf8string2wstring(word_list[0]).length());
            finals_list.push_back(finals_temp);
            finals_temp.assign(
                finals.begin() +
                    ppspeech::utf8string2wstring(word_list[0]).length(),
                finals.end());
            finals_list.push_back(finals_temp);

            finals = {};
            for (int i = 0; i < finals_list.size(); i++) {
                // e.g. 所有/人
                if (AllToneThree(finals_list[i]) &&
                    finals_list[i].size() == 2) {
                    finals_list[i][0] = finals_list[i][0].replace(
                        finals_list[i][0].length() - 1, 1, "2");
                } else if (i == 1 && !(AllToneThree(finals_list[i])) &&
                           absl::EndsWith(finals_list[i][0], "3") == true &&
                           absl::EndsWith(finals_list[0].back(), "3") == true) {
                    finals_list[0].back() = finals_list[0].back().replace(
                        finals_list[0].back().length() - 1, 1, "2");
                }
            }
            finals.insert(
                finals.end(), finals_list[0].begin(), finals_list[0].end());
            finals.insert(
                finals.end(), finals_list[1].begin(), finals_list[1].end());
        }

    } else if (word_num == 4) {  //将成语拆分为两个长度为 2 的单词
        // 创建对应的 韵母列表
        finals_temp = {};
        finals_list = {};
        finals_temp.assign(finals.begin(), finals.begin() + 2);
        finals_list.push_back(finals_temp);
        finals_temp.assign(finals.begin() + 2, finals.end());
        finals_list.push_back(finals_temp);

        finals = {};
        for (int j = 0; j < finals_list.size(); j++) {
            if (AllToneThree(finals_list[j])) {
                finals_list[j][0] = finals_list[j][0].replace(
                    finals_list[j][0].length() - 1, 1, "2");
            }
            finals.insert(
                finals.end(), finals_list[j].begin(), finals_list[j].end());
        }
    }

    return 0;
}

int FrontEngineInterface::ModifyTone(const std::string &word,
                                     const std::string &pos,
                                     std::vector<std::string> &finals) {
    if ((0 != BuSandi(word, finals)) || (0 != YiSandhi(word, finals)) ||
        (0 != NeuralSandhi(word, pos, finals)) ||
        (0 != ThreeSandhi(word, finals))) {
        LOG(ERROR) << "Failed to modify tone of the word: " << word;
        return -1;
    }

    return 0;
}

std::vector<std::vector<std::string>> FrontEngineInterface::MergeErhua(
    const std::vector<std::string> &initials,
    const std::vector<std::string> &finals,
    const std::string &word,
    const std::string &pos) {
    std::vector<std::string> new_initials = {};
    std::vector<std::string> new_finals = {};
    std::vector<std::vector<std::string>> new_initials_finals;
    std::vector<std::string> specified_pos = {"a", "j", "nr"};
    std::wstring word_wstr = ppspeech::utf8string2wstring(word);
    std::vector<std::wstring> wordvec;
    // 一个词转成向量形式
    if (0 != Word2WordVec(word, wordvec)) {
        LOG(ERROR) << "Failed to get word vector";
    }
    int word_num = wordvec.size();

    if ((find(must_erhua.begin(), must_erhua.end(), word) ==
         must_erhua.end()) &&
        ((find(not_erhua.begin(), not_erhua.end(), word) != not_erhua.end()) ||
         (find(specified_pos.begin(), specified_pos.end(), pos) !=
          specified_pos.end()))) {
        new_initials_finals.push_back(initials);
        new_initials_finals.push_back(finals);
        return new_initials_finals;
    }
    if (finals.size() != word_num) {
        new_initials_finals.push_back(initials);
        new_initials_finals.push_back(finals);
        return new_initials_finals;
    }

    assert(finals.size() == word_num);
    for (int i = 0; i < finals.size(); i++) {
        if (i == finals.size() - 1 && wordvec[i] == L"儿" &&
            (finals[i] == "er2" || finals[i] == "er5") && word_num >= 2 &&
            find(not_erhua.begin(),
                 not_erhua.end(),
                 ppspeech::wstring2utf8string(word_wstr.substr(
                     word_wstr.length() - 2))) == not_erhua.end() &&
            !new_finals.empty()) {
            new_finals.back() =
                new_finals.back().substr(0, new_finals.back().length() - 1) +
                "r" + new_finals.back().substr(new_finals.back().length() - 1);
        } else {
            new_initials.push_back(initials[i]);
            new_finals.push_back(finals[i]);
        }
    }
    new_initials_finals.push_back(new_initials);
    new_initials_finals.push_back(new_finals);

    return new_initials_finals;
}
}  // namespace ppspeech
