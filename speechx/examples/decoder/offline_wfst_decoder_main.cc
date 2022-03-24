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

// todo refactor, repalce with gtest

#include "base/flags.h"
#include "base/log.h"
#include "decoder/ctc_tlg_decoder.h"
#include "frontend/raw_audio.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"
#include "nnet/paddle_nnet.h"

DEFINE_string(feature_respecifier, "", "test feature rspecifier");
DEFINE_string(model_path, "avg_1.jit.pdmodel", "paddle nnet model");
DEFINE_string(param_path, "avg_1.jit.pdiparams", "paddle nnet model param");
DEFINE_string(word_symbol_table, "vocab.txt", "word symbol table");
DEFINE_string(graph_path, "TLG", "decoder graph");


using kaldi::BaseFloat;
using kaldi::Matrix;
using std::vector;

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    kaldi::SequentialBaseFloatMatrixReader feature_reader(
        FLAGS_feature_respecifier);
    std::string model_graph = FLAGS_model_path;
    std::string model_params = FLAGS_param_path;
    std::string word_symbol_table = FLAGS_word_symbol_table;
    std::string graph_path = FLAGS_graph_path;

    int32 num_done = 0, num_err = 0;

    ppspeech::TLGDecoderOptions opts;
    opts.word_symbol_table = word_symbol_table;
    opts.fst_path = graph_path;
    ppspeech::TLGDecoder decoder(opts);

    ppspeech::ModelOptions model_opts;
    model_opts.model_path = model_graph;
    model_opts.params_path = model_params;
    std::shared_ptr<ppspeech::PaddleNnet> nnet(
        new ppspeech::PaddleNnet(model_opts));
    std::shared_ptr<ppspeech::RawDataCache> raw_data(
        new ppspeech::RawDataCache());
    std::shared_ptr<ppspeech::Decodable> decodable(
        new ppspeech::Decodable(nnet, raw_data));

    int32 chunk_size = 35;
    decoder.InitDecoder();

    for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        const kaldi::Matrix<BaseFloat> feature = feature_reader.Value();
        raw_data->SetDim(feature.NumCols());
        int32 row_idx = 0;
        int32 num_chunks = feature.NumRows() / chunk_size;
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            kaldi::Vector<kaldi::BaseFloat> feature_chunk(chunk_size *
                                                          feature.NumCols());
            for (int row_id = 0; row_id < chunk_size; ++row_id) {
                kaldi::SubVector<kaldi::BaseFloat> tmp(feature, row_idx);
                kaldi::SubVector<kaldi::BaseFloat> f_chunk_tmp(
                    feature_chunk.Data() + row_id * feature.NumCols(),
                    feature.NumCols());
                f_chunk_tmp.CopyFromVec(tmp);
                row_idx++;
            }
            raw_data->Accept(feature_chunk);
            if (chunk_idx == num_chunks - 1) {
                raw_data->SetFinished();
            }
            decoder.AdvanceDecode(decodable);
        }
        std::string result;
        result = decoder.GetFinalBestPath();
        KALDI_LOG << " the result of " << utt << " is " << result;
        decodable->Reset();
        decoder.Reset();
        ++num_done;
    }

    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
