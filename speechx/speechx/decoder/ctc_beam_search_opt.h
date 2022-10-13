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
    CTCBeamSearchOptions()
        : blank(0),
          dict_file("vocab.txt"),
          lm_path(""),
          alpha(1.9f),
          beta(5.0),
          beam_size(300),
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


// used by u2 model
struct CTCBeamSearchDecoderOptions {
    // chunk_size is the frame number of one chunk after subsampling.
    // e.g. if subsample rate is 4 and chunk_size = 16, the frames in
    // one chunk are 67=16*4 + 3, stride is 64=16*4
    int chunk_size;
    int num_left_chunks;

    // final_score = rescoring_weight * rescoring_score + ctc_weight *
    // ctc_score;
    // rescoring_score = left_to_right_score * (1 - reverse_weight) +
    // right_to_left_score * reverse_weight
    // Please note the concept of ctc_scores
    // in the following two search methods are different. For
    // CtcPrefixBeamSerch,
    // it's a sum(prefix) score + context score For CtcWfstBeamSerch, it's a
    // max(viterbi) path score + context score So we should carefully set
    // ctc_weight accroding to the search methods.
    float ctc_weight;
    float rescoring_weight;
    float reverse_weight;

    // CtcEndpointConfig ctc_endpoint_opts;

    CTCBeamSearchOptions ctc_prefix_search_opts;

    CTCBeamSearchDecoderOptions()
        : chunk_size(16),
          num_left_chunks(-1),
          ctc_weight(0.5),
          rescoring_weight(1.0),
          reverse_weight(0.0) {}

    void Register(kaldi::OptionsItf* opts) {
        std::string module = "DecoderConfig: ";
        opts->Register(
            "chunk-size",
            &chunk_size,
            module + "the frame number of one chunk after subsampling.");
        opts->Register("num-left-chunks",
                       &num_left_chunks,
                       module + "the left history chunks number.");
        opts->Register("ctc-weight",
                       &ctc_weight,
                       module +
                           "ctc weight for rescore. final_score = "
                           "rescoring_weight * rescoring_score + ctc_weight * "
                           "ctc_score.");
        opts->Register("rescoring-weight",
                       &rescoring_weight,
                       module +
                           "attention score weight for rescore. final_score = "
                           "rescoring_weight * rescoring_score + ctc_weight * "
                           "ctc_score.");
        opts->Register("reverse-weight",
                       &reverse_weight,
                       module +
                           "reverse decoder weight. rescoring_score = "
                           "left_to_right_score * (1 - reverse_weight) + "
                           "right_to_left_score * reverse_weight.");
    }
};

}  // namespace ppspeech