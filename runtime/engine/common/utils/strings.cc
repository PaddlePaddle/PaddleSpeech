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

}  // namespace ppspeech