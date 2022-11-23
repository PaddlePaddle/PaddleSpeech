// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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

#pragma once

#include "base/common.h"

struct DecoderResult {
    BaseFloat acoustic_score;
    std::vector<int32> words_idx;
    std::vector<std::pair<int32, int32>> time_stamp;
};


namespace ppspeech {

struct WordPiece {
    std::string word;
    int start = -1;
    int end = -1;

    WordPiece(std::string word, int start, int end)
        : word(std::move(word)), start(start), end(end) {}
};

struct DecodeResult {
    float score = -kBaseFloatMax;
    std::string sentence;
    std::vector<WordPiece> word_pieces;

    static bool CompareFunc(const DecodeResult& a, const DecodeResult& b) {
        return a.score > b.score;
    }
};

}  // namespace ppspeech
