// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>

#include "utils/strings.h"

namespace ppspeech {

std::vector<std::string> StrSplit(const std::string& str,
                                  const char* delim,
                                  bool omit_empty_string) {
    std::vector<std::string> outs;
    int start = 0;
    int end = str.size();
    int found = 0;
    while (found != std::string::npos) {
        found = str.find_first_of(delim, start);
        // start != end condition is for when the delimiter is at the end
        if (!omit_empty_string || (found != start && start != end)) {
            outs.push_back(str.substr(start, found - start));
        }
        start = found + 1;
    }

    return outs;
}


std::string StrJoin(const std::vector<std::string>& strs, const char* delim) {
    std::stringstream ss;
    for (ssize_t i = 0; i < strs.size(); ++i) {
        ss << strs[i];
        if (i < strs.size() - 1) {
            ss << std::string(delim);
        }
    }
    return ss.str();
}

std::string DelBlank(const std::string& str) {
    std::string out = "";
    int ptr_in = 0;    //  the pointer of input string (for traversal)
    int end = str.size();
    int ptr_out = -1;  //  the pointer of output string (last char)
    while (ptr_in != end) {
        while (ptr_in != end && str[ptr_in] == ' ') {
            ptr_in += 1;
        }
        if (ptr_in == end) 
            return out;
        if (ptr_out != -1 && isalpha(str[ptr_in]) && isalpha(str[ptr_out]) && str[ptr_in-1] == ' ')
            // add a space when the last and current chars are in English and there have space(s) between them
            out += ' ';
        out += str[ptr_in];
        ptr_out = ptr_in;
        ptr_in += 1;
    }
    return out;
}

std::string AddBlank(const std::string& str) {
    std::string out = "";
    int ptr = 0;  // the pointer of the input string
    int end = str.size();
    while (ptr != end) {
        if (isalpha(str[ptr])) {
            if (ptr == 0 or str[ptr-1] != ' ')
                out += " ";  // add pre-space for an English word
            while (isalpha(str[ptr])) {
                out += str[ptr];
                ptr += 1;
            }
            out += " ";  // add post-space for an English word
        } else {
            out += str[ptr];
            ptr += 1;
        }
    }
    return out;
}

std::string ReverseFraction(const std::string& str) {
    std::string out = "";
    int ptr = 0;   // the pointer of the input string
    int end = str.size();
    int left, right, frac;  // the start index of the left tag, right tag and '/'.
    left = right = frac = 0;
    int len_tag = 5;  // length of "<tag>"

    while (ptr != end) {
        // find the position of left tag, right tag and '/'. (xxx<tag>num1/num2</tag>)
        left = str.find("<tag>", ptr);
        if (left == -1)
            break;
        out += str.substr(ptr, left - ptr);  // content before left tag (xxx)
        frac = str.find("/", left);
        right = str.find("<tag>", frac);
        
        out += str.substr(frac + 1, right - frac - 1) + '/' + 
               str.substr(left + len_tag, frac - left - len_tag);  // num2/num1
        ptr = right + len_tag;
    }
    if (ptr != end) {
        out += str.substr(ptr, end - ptr);
    }
    return out;
}

#ifdef _MSC_VER
std::wstring ToWString(const std::string& str) {
    unsigned len = str.size() * 2;
    setlocale(LC_CTYPE, "");
    wchar_t* p = new wchar_t[len];
    mbstowcs(p, str.c_str(), len);
    std::wstring wstr(p);
    delete[] p;
    return wstr;
}
#endif

}  // namespace ppspeech
