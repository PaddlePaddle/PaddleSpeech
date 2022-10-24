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

#include "base/common.h"
#include "util/parse-options.h"

#pragma once

namespace ppspeech {


struct CTCBeamSearchOptions {
    // common
    int blank;

    // ds2
    std::string dict_file;
    std::string lm_path;
    int beam_size;
    BaseFloat alpha;
    BaseFloat beta;
    BaseFloat cutoff_prob;
    int cutoff_top_n;
    int num_proc_bsearch;

    // u2
    int first_beam_size;
    int second_beam_size;
    explicit CTCBeamSearchOptions()
        : blank(0),
          dict_file("vocab.txt"),
          lm_path(""),
          beam_size(300),
          alpha(1.9f),
          beta(5.0),
          cutoff_prob(0.99f),
          cutoff_top_n(40),
          num_proc_bsearch(10),
          first_beam_size(10),
          second_beam_size(10) {}

    void Register(kaldi::OptionsItf* opts) {
        std::string module = "Ds2BeamSearchConfig: ";
        opts->Register("dict", &dict_file, module + "vocab file path.");
        opts->Register(
            "lm-path", &lm_path, module + "ngram language model path.");
        opts->Register("alpha", &alpha, module + "alpha");
        opts->Register("beta", &beta, module + "beta");
        opts->Register("beam-size",
                       &beam_size,
                       module + "beam size for beam search method");
        opts->Register("cutoff-prob", &cutoff_prob, module + "cutoff probs");
        opts->Register("cutoff-top-n", &cutoff_top_n, module + "cutoff top n");
        opts->Register(
            "num-proc-bsearch", &num_proc_bsearch, module + "num proc bsearch");

        opts->Register("blank", &blank, "blank id, default is 0.");

        module = "U2BeamSearchConfig: ";
        opts->Register(
            "first-beam-size", &first_beam_size, module + "first beam size.");
        opts->Register("second-beam-size",
                       &second_beam_size,
                       module + "second beam size.");
    }
};

}  // namespace ppspeech