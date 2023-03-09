#include "front/text_normalize.h"

namespace speechnn {

// 初始化 digits_map and unit_map
int TextNormalizer::InitMap() {
   
    digits_map["0"] = "零";
    digits_map["1"] = "一";
    digits_map["2"] = "二";
    digits_map["3"] = "三";
    digits_map["4"] = "四";
    digits_map["5"] = "五";
    digits_map["6"] = "六";
    digits_map["7"] = "七";
    digits_map["8"] = "八";
    digits_map["9"] = "九";

    units_map[1] = "十";
    units_map[2] = "百";
    units_map[3] = "千";
    units_map[4] = "万";
    units_map[8] = "亿";
   
    return 0;
}

// 替换
int TextNormalizer::Replace(std::wstring &sentence, const int &pos, const int &len, const std::wstring &repstr) {
    // 删除原来的
    sentence.erase(pos, len);
    // 插入新的
    sentence.insert(pos, repstr);
    return 0;

}

// 根据标点符号切分句子
int TextNormalizer::SplitByPunc(const std::wstring &sentence, std::vector<std::wstring> &sentence_part) {
    std::wstring temp = sentence;
    std::wregex reg(L"[：，；。？！,;?!]");
    std::wsmatch match;

    while (std::regex_search (temp, match, reg)) {
        sentence_part.push_back(temp.substr(0, match.position(0) + match.length(0)));
        Replace(temp, 0, match.position(0) + match.length(0), L"");
    }
    // 如果最后没有标点符号
    if(temp != L"") {
        sentence_part.push_back(temp);
    }
    return 0;
}

//数字转文本，10200 - > 一万零二百
std::string TextNormalizer::CreateTextValue(const std::string &num_str, bool use_zero) {

    std::string num_lstrip = std::string(absl::StripPrefix(num_str, "0")).data();
    int len = num_lstrip.length();
    
    if(len == 0) {
        return "";
    } else if (len == 1) {
        if(use_zero && (len < num_str.length())) {
            return digits_map["0"] + digits_map[num_lstrip];
        } else {
            return digits_map[num_lstrip];
        }
    } else {
        int largest_unit = 0; // 最大单位
        std::string first_part;
        std::string second_part;

        if (len > 1 and len <= 2) {
            largest_unit = 1;
        } else if (len > 2 and len <= 3) {
            largest_unit = 2;
        } else if (len > 3 and len <= 4) {
            largest_unit = 3;
        } else if (len > 4 and len <= 8) {
            largest_unit = 4;
        } else if (len > 8) {
            largest_unit = 8;  
        }  

        first_part = num_str.substr(0, num_str.length() - largest_unit);
        second_part = num_str.substr(num_str.length() - largest_unit);
        
        return CreateTextValue(first_part, use_zero) + units_map[largest_unit] + CreateTextValue(second_part, use_zero);
    }
}

//  数字一个一个对应，可直接用于年份，电话，手机，
std::string TextNormalizer::SingleDigit2Text(const std::string &num_str, bool alt_one) {
    std::string text = "";
    if (alt_one) {
        digits_map["1"] = "幺";
    } else {
        digits_map["1"] = "一";
    }

    for (size_t i = 0; i < num_str.size(); i++) {
        std::string num_int(1, num_str[i]);
        if (digits_map.find(num_int) == digits_map.end()) {
            LOG(ERROR) << "digits_map doesn't have key: " << num_int;
        }
        text += digits_map[num_int];
    }

    return text;
}

std::string TextNormalizer::SingleDigit2Text(const std::wstring &num, bool alt_one) {
    std::string num_str = wstring2utf8string(num);
    return SingleDigit2Text(num_str, alt_one);
}

//  数字整体对应，可直接用于月份，日期，数值整数部分
std::string TextNormalizer::MultiDigit2Text(const std::string &num_str, bool alt_one, bool use_zero) {
    LOG(INFO) << "aaaaaaaaaaaaaaaa: " << alt_one << use_zero;
    if (alt_one) {
        digits_map["1"] = "幺";
    } else {
        digits_map["1"] = "一";
    }

    std::wstring result = utf8string2wstring(CreateTextValue(num_str, use_zero));
    std::wstring result_0(1, result[0]);
    std::wstring result_1(1, result[1]);
    // 一十八 --> 十八
    if ((result_0 == utf8string2wstring(digits_map["1"])) && (result_1 == utf8string2wstring(units_map[1]))) {
        return wstring2utf8string(result.substr(1,result.length()));
    } else {
        return wstring2utf8string(result);
    }
}

std::string TextNormalizer::MultiDigit2Text(const std::wstring &num, bool alt_one, bool use_zero) {
    std::string num_str = wstring2utf8string(num);
    return MultiDigit2Text(num_str, alt_one, use_zero);
}

// 数字转文本，包括整数和小数
std::string TextNormalizer::Digits2Text(const std::string &num_str) {
    std::string text;
    std::vector<std::string> integer_decimal;
    integer_decimal = absl::StrSplit(num_str, ".");
    
    if(integer_decimal.size() == 1) {  // 整数
        text = MultiDigit2Text(integer_decimal[0]);
    } else if(integer_decimal.size() == 2) {   // 小数
        if(integer_decimal[0] == "") {  // 无整数的小数类型，例如：.22
            text = "点" + SingleDigit2Text(std::string(absl::StripSuffix(integer_decimal[1], "0")).data());
        } else {  // 常规小数类型，例如：12.34
            text = MultiDigit2Text(integer_decimal[0]) + "点" + \
                   SingleDigit2Text(std::string(absl::StripSuffix(integer_decimal[1], "0")).data());
        }
    } else {
        return "The value does not conform to the numeric format";
    }

    return text;
}

std::string TextNormalizer::Digits2Text(const std::wstring &num) {
    std::string num_str = wstring2utf8string(num);
    return Digits2Text(num_str);
}

// 日期，2021年8月18日 --> 二零二一年八月十八日
int TextNormalizer::ReData(std::wstring &sentence) {
    std::wregex reg(L"(\\d{4}|\\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?");
    std::wsmatch match;
    std::string rep;

    while (std::regex_search (sentence, match, reg)) {
        rep = "";
        rep += SingleDigit2Text(match[1]) + "年";
        if(match[3] != L"") {
            rep += MultiDigit2Text(match[3], false, false) + "月";
        }
        if(match[5] != L"") {
            rep += MultiDigit2Text(match[5], false, false) + wstring2utf8string(match[9]);
        }

        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));

    }

    return 0;
}


// XX-XX-XX or XX/XX/XX 例如：2021/08/18 --> 二零二一年八月十八日
int TextNormalizer::ReData2(std::wstring &sentence) {
    std::wregex reg(L"(\\d{4})([- /.])(0[1-9]|1[012])\\2(0[1-9]|[12][0-9]|3[01])");
    std::wsmatch match;
    std::string rep;
     
    while (std::regex_search (sentence, match, reg)) {
        rep = "";
        rep += (SingleDigit2Text(match[1]) + "年");
        rep += (MultiDigit2Text(match[3], false, false) + "月");
        rep += (MultiDigit2Text(match[4], false, false) + "日");
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));

    }
    
    return 0;
}

// XX:XX:XX   09:09:02 --> 九点零九分零二秒
int TextNormalizer::ReTime(std::wstring &sentence) {
    std::wregex reg(L"([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?");
    std::wsmatch match;
    std::string rep;
    
    while (std::regex_search (sentence, match, reg)) {
        rep = "";
        rep += (MultiDigit2Text(match[1], false, false) + "点");
        if(absl::StartsWith(wstring2utf8string(match[2]), "0")) {
            rep += "零";
        }
        rep += (MultiDigit2Text(match[2]) + "分");
        if(absl::StartsWith(wstring2utf8string(match[4]), "0")) {
            rep += "零";
        }
        rep += (MultiDigit2Text(match[4]) + "秒");

        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));
    }

    return 0;
}

// 温度，例如：-24.3℃ --> 零下二十四点三度
int TextNormalizer::ReTemperature(std::wstring &sentence) {
    std::wregex reg(L"(-?)(\\d+(\\.\\d+)?)(°C|℃|度|摄氏度)"); 
    std::wsmatch match;
    std::string rep;
    std::string sign;
    std::vector<std::string> integer_decimal;
    std::string unit;

    while (std::regex_search (sentence, match, reg)) {
        match[1] == L"-" ? sign = "负" : sign = "";
        match[4] == L"摄氏度"? unit = "摄氏度" : unit = "度";
        rep = sign + Digits2Text(match[2]) + unit;
        
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));

    }

    return 0;

}

// 分数，例如： 1/3 --> 三分之一
int TextNormalizer::ReFrac(std::wstring &sentence) {
    std::wregex reg(L"(-?)(\\d+)/(\\d+)"); 
    std::wsmatch match;
    std::string sign;
    std::string rep;
    while (std::regex_search (sentence, match, reg)) {
        match[1] == L"-" ? sign = "负" : sign = "";
        rep = sign + MultiDigit2Text(match[3]) + "分之" + MultiDigit2Text(match[2]);
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));
    }

    return 0;
}

// 百分数，例如：45.5% --> 百分之四十五点五
int TextNormalizer::RePercentage(std::wstring &sentence) {
    std::wregex reg(L"(-?)(\\d+(\\.\\d+)?)%"); 
    std::wsmatch match;
    std::string sign;
    std::string rep;
    std::vector<std::string> integer_decimal;
    
    while (std::regex_search (sentence, match, reg)) {
        match[1] == L"-" ? sign = "负" : sign = "";
        rep = sign + "百分之" + Digits2Text(match[2]);
         
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));
    }
    
    return 0;
}

// 手机号码，例如：+86 18883862235 --> 八六幺八八八三八六二二三五
int TextNormalizer::ReMobilePhone(std::wstring &sentence) {
    std::wregex reg(L"(\\d)?((\\+?86 ?)?1([38]\\d|5[0-35-9]|7[678]|9[89])\\d{8})(\\d)?");
    std::wsmatch match;
    std::string rep;
    std::vector<std::string> country_phonenum;

    while (std::regex_search (sentence, match, reg)) {
        country_phonenum = absl::StrSplit(wstring2utf8string(match[0]), "+");
        rep = "";
        for(int i = 0; i < country_phonenum.size(); i++) {
            LOG(INFO) << country_phonenum[i];
            rep += SingleDigit2Text(country_phonenum[i], true);
        }
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));

    }
    
    return 0;
}

// 座机号码，例如：010-51093154 --> 零幺零五幺零九三幺五四
int TextNormalizer::RePhone(std::wstring &sentence) {
    std::wregex reg(L"(\\d)?((0(10|2[1-3]|[3-9]\\d{2})-?)?[1-9]\\d{6,7})(\\d)?");
    std::wsmatch match;
    std::vector<std::string> zone_phonenum;
    std::string rep;

    while (std::regex_search (sentence, match, reg)) {
        rep = "";
        zone_phonenum = absl::StrSplit(wstring2utf8string(match[0]), "-");
        for(int i = 0; i < zone_phonenum.size(); i ++) {
            rep += SingleDigit2Text(zone_phonenum[i], true);
        }
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));
    }

    return 0;
}

// 范围，例如：60~90 --> 六十到九十
int TextNormalizer::ReRange(std::wstring &sentence) {
    std::wregex reg(L"((-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+)))[-~]((-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+)))");
    std::wsmatch match;
    std::string rep;
    std::string sign1;
    std::string sign2;

    while (std::regex_search (sentence, match, reg)) {
        rep = "";
        match[2] == L"-" ? sign1 = "负" : sign1 = "";
        if(match[6] != L"") {
            rep += sign1 + Digits2Text(match[6]) + "到";
        } else {
            rep += sign1 + Digits2Text(match[3]) + "到";
        }
        match[9] == L"-" ? sign2 = "负" : sign2 = "";
        if(match[13] != L"") {
            rep += sign2 + Digits2Text(match[13]);
        } else {
            rep += sign2 + Digits2Text(match[10]);
        }

        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));
    }

    return 0;
}

// 带负号的整数，例如：-10 --> 负十
int TextNormalizer::ReInterger(std::wstring &sentence) {
    std::wregex reg(L"(-)(\\d+)"); 
    std::wsmatch match;
    std::string rep;
    while (std::regex_search (sentence, match, reg)) {
        rep = "负" + MultiDigit2Text(match[2]);
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));
    }
    
    return 0;
}

// 纯小数
int TextNormalizer::ReDecimalNum(std::wstring &sentence) {
    std::wregex reg(L"(-?)((\\d+)(\\.\\d+))|(\\.(\\d+))"); 
    std::wsmatch match;
    std::string sign;
    std::string rep;
    //std::vector<std::string> integer_decimal;
    while (std::regex_search (sentence, match, reg)) {
        match[1] == L"-" ? sign = "负" : sign = "";
        if(match[5] != L"") {
            rep = sign + Digits2Text(match[5]);
        } else {
            rep = sign + Digits2Text(match[2]);
        }

        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));
    }

    return 0;
}

// 正整数 + 量词
int TextNormalizer::RePositiveQuantifiers(std::wstring &sentence) {
    std::wstring common_quantifiers = L"(朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲| \
    墙|群|腔|砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|针|线|管|名|位|身|堂| \
    课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘| \
    毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日| \
    季|刻|时|周|天|秒|分|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万| \
    万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|)块|角|毛|分)";
    std::wregex reg(L"(\\d+)([多余几])?" + common_quantifiers); 
    std::wsmatch match;
    std::string rep;
    while (std::regex_search (sentence, match, reg)) {
        rep = MultiDigit2Text(match[1]);
        Replace(sentence, match.position(1), match.length(1), utf8string2wstring(rep));
    }

    return 0;
}

// 编号类数字，例如： 89757 --> 八九七五七
int TextNormalizer::ReDefalutNum(std::wstring &sentence) {
    std::wregex reg(L"\\d{3}\\d*"); 
    std::wsmatch match;
    while (std::regex_search (sentence, match, reg)) {
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(SingleDigit2Text(match[0])));
    }

    return 0;
}

int TextNormalizer::ReNumber(std::wstring &sentence) {
    std::wregex reg(L"(-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+))"); 
    std::wsmatch match;
    std::string sign;
    std::string rep;
    while (std::regex_search (sentence, match, reg)) {
        match[1] == L"-" ? sign = "负" : sign = "";
        if(match[5] != L"") {
            rep = sign + Digits2Text(match[5]);
        } else {
            rep = sign + Digits2Text(match[2]);
        }
        
        Replace(sentence, match.position(0), match.length(0), utf8string2wstring(rep));
    }
    return 0;
}

// 整体正则，按顺序
int TextNormalizer::SentenceNormalize(std::wstring &sentence) {
    ReData(sentence);
    ReData2(sentence);
    ReTime(sentence);
    ReTemperature(sentence);
    ReFrac(sentence);
    RePercentage(sentence);
    ReMobilePhone(sentence);
    RePhone(sentence);
    ReRange(sentence);
    ReInterger(sentence);
    ReDecimalNum(sentence);
    RePositiveQuantifiers(sentence);  
    ReDefalutNum(sentence);
    ReNumber(sentence);
    return 0;   
}


}