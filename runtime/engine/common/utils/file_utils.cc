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

#include "utils/file_utils.h"

namespace ppspeech {

bool ReadFileToVector(const std::string& filename,
                      std::vector<std::string>* vocabulary) {
    std::ifstream file_in(filename);
    if (!file_in) {
        std::cerr << "please input a valid file" << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file_in, line)) {
        vocabulary->emplace_back(line);
    }

    return true;
}

std::string ReadFile2String(const std::string& path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cerr << "please input a valid file" << std::endl;
    }
    return std::string((std::istreambuf_iterator<char>(input_file)),
                       std::istreambuf_iterator<char>());
}
}  // namespace ppspeech
