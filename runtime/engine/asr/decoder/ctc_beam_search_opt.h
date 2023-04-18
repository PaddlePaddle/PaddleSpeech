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
#include "util/parse-options.h"

namespace ppspeech {


struct CTCBeamSearchOptions {
    // common
    int blank;
    std::string word_symbol_table;

    // u2
    int first_beam_size;
    int second_beam_size;
    
    CTCBeamSearchOptions()
        : blank(0),
          word_symbol_table("vocab.txt"),
          first_beam_size(10),
          second_beam_size(10) {}

    void Register(kaldi::OptionsItf* opts) {
        std::string module = "CTCBeamSearchOptions: ";
        opts->Register("word_symbol_table", &word_symbol_table, module + "vocab file path.");
        opts->Register("blank", &blank, "blank id, default is 0.");
        opts->Register(
            "first-beam-size", &first_beam_size, module + "first beam size.");
        opts->Register("second-beam-size",
                       &second_beam_size,
                       module + "second beam size.");
    }
};

}  // namespace ppspeech
