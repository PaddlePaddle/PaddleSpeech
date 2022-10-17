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
#include "decoder/ctc_beam_search_decoder.h"
#include "decoder/ctc_tlg_decoder.h"

// feature
DEFINE_bool(use_fbank, false, "False for fbank; or linear feature");
// DEFINE_bool(to_float32, true, "audio convert to pcm32. True for linear
// feature, or fbank");
DEFINE_int32(num_bins, 161, "num bins of mel");
DEFINE_string(cmvn_file, "", "read cmvn");

// feature sliding window
DEFINE_int32(receptive_field_length,
             7,
             "receptive field of two CNN(kernel=3) downsampling module.");
DEFINE_int32(subsampling_rate,
             4, 
             "two CNN(kernel=3) module downsampling rate.");
DEFINE_int32(nnet_decoder_chunk, 1, "paddle nnet forward chunk");


// nnet
DEFINE_string(vocab_path, "", "nnet vocab path.");
DEFINE_string(model_path, "avg_1.jit.pdmodel", "paddle nnet model");
DEFINE_string(param_path, "avg_1.jit.pdiparams", "paddle nnet model param");
DEFINE_string(
    model_input_names,
    "audio_chunk,audio_chunk_lens,chunk_state_h_box,chunk_state_c_box",
    "model input names");
DEFINE_string(model_output_names,
              "softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0",
              "model output names");
DEFINE_string(model_cache_names,
              "chunk_state_h_box,chunk_state_c_box",
              "model cache names");
DEFINE_string(model_cache_shapes, "5-1-1024,5-1-1024", "model cache shapes");


// decoder
DEFINE_double(acoustic_scale, 1.0, "acoustic scale");

DEFINE_string(graph_path, "TLG", "decoder graph");
DEFINE_string(word_symbol_table, "words.txt", "word symbol table");
DEFINE_int32(max_active, 7500, "max active");
DEFINE_double(beam, 15.0, "decoder beam");
DEFINE_double(lattice_beam, 7.5, "decoder beam");



// DecodeOptions flags
// DEFINE_int32(chunk_size, -1, "decoding chunk size");
DEFINE_int32(num_left_chunks, -1, "left chunks in decoding");
DEFINE_double(ctc_weight,
              0.5,
              "ctc weight when combining ctc score and rescoring score");
DEFINE_double(rescoring_weight,
              1.0,
              "rescoring weight when combining ctc score and rescoring score");
DEFINE_double(reverse_weight,
              0.3,
              "used for bitransformer rescoring. it must be 0.0 if decoder is"
              "conventional transformer decoder, and only reverse_weight > 0.0"
              "dose the right to left decoder will be calculated and used");
DEFINE_int32(nbest, 10, "nbest for ctc wfst or prefix search");
DEFINE_int32(blank, 0, "blank id in vocab");
