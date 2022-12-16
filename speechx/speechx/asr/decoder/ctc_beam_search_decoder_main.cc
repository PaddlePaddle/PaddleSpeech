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

// used by deepspeech2

#include "base/flags.h"
#include "base/log.h"
#include "decoder/ctc_beam_search_decoder.h"
#include "frontend/audio/data_cache.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"
#include "nnet/ds2_nnet.h"

DEFINE_string(feature_rspecifier, "", "test feature rspecifier");
DEFINE_string(result_wspecifier, "", "test result wspecifier");
DEFINE_string(model_path, "avg_1.jit.pdmodel", "paddle nnet model");
DEFINE_string(param_path, "avg_1.jit.pdiparams", "paddle nnet model param");
DEFINE_string(dict_file, "vocab.txt", "vocabulary of lm");
DEFINE_string(lm_path, "", "language model");
DEFINE_int32(receptive_field_length,
             7,
             "receptive field of two CNN(kernel=3) downsampling module.");
DEFINE_int32(subsampling_rate,
             4,
             "two CNN(kernel=3) module downsampling rate.");
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
DEFINE_int32(nnet_decoder_chunk, 1, "paddle nnet forward chunk");

using kaldi::BaseFloat;
using kaldi::Matrix;
using std::vector;

// test ds2 online decoder by feeding speech feature
int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    CHECK_NE(FLAGS_result_wspecifier, "");
    CHECK_NE(FLAGS_feature_rspecifier, "");

    kaldi::SequentialBaseFloatMatrixReader feature_reader(
        FLAGS_feature_rspecifier);
    kaldi::TokenWriter result_writer(FLAGS_result_wspecifier);
    std::string model_path = FLAGS_model_path;
    std::string model_params = FLAGS_param_path;
    std::string dict_file = FLAGS_dict_file;
    std::string lm_path = FLAGS_lm_path;
    LOG(INFO) << "model path: " << model_path;
    LOG(INFO) << "model param: " << model_params;
    LOG(INFO) << "dict path: " << dict_file;
    LOG(INFO) << "lm path: " << lm_path;

    int32 num_done = 0, num_err = 0;

    ppspeech::CTCBeamSearchOptions opts;
    opts.dict_file = dict_file;
    opts.lm_path = lm_path;
    ppspeech::CTCBeamSearch decoder(opts);

    ppspeech::ModelOptions model_opts = ppspeech::ModelOptions::InitFromFlags();

    std::shared_ptr<ppspeech::PaddleNnet> nnet(
        new ppspeech::PaddleNnet(model_opts));
    std::shared_ptr<ppspeech::DataCache> raw_data(new ppspeech::DataCache());
    std::shared_ptr<ppspeech::Decodable> decodable(
        new ppspeech::Decodable(nnet, raw_data));

    int32 chunk_size = FLAGS_receptive_field_length +
                       (FLAGS_nnet_decoder_chunk - 1) * FLAGS_subsampling_rate;
    int32 chunk_stride = FLAGS_subsampling_rate * FLAGS_nnet_decoder_chunk;
    int32 receptive_field_length = FLAGS_receptive_field_length;
    LOG(INFO) << "chunk size (frame): " << chunk_size;
    LOG(INFO) << "chunk stride (frame): " << chunk_stride;
    LOG(INFO) << "receptive field (frame): " << receptive_field_length;
    decoder.InitDecoder();

    kaldi::Timer timer;
    for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        kaldi::Matrix<BaseFloat> feature = feature_reader.Value();
        raw_data->SetDim(feature.NumCols());
        LOG(INFO) << "process utt: " << utt;
        LOG(INFO) << "rows: " << feature.NumRows();
        LOG(INFO) << "cols: " << feature.NumCols();

        int32 row_idx = 0;
        int32 padding_len = 0;
        int32 ori_feature_len = feature.NumRows();
        if ((feature.NumRows() - chunk_size) % chunk_stride != 0) {
            padding_len =
                chunk_stride - (feature.NumRows() - chunk_size) % chunk_stride;
            feature.Resize(feature.NumRows() + padding_len,
                           feature.NumCols(),
                           kaldi::kCopyData);
        }
        int32 num_chunks = (feature.NumRows() - chunk_size) / chunk_stride + 1;
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            kaldi::Vector<kaldi::BaseFloat> feature_chunk(chunk_size *
                                                          feature.NumCols());
            int32 feature_chunk_size = 0;
            if (ori_feature_len > chunk_idx * chunk_stride) {
                feature_chunk_size = std::min(
                    ori_feature_len - chunk_idx * chunk_stride, chunk_size);
            }
            if (feature_chunk_size < receptive_field_length) break;

            int32 start = chunk_idx * chunk_stride;

            for (int row_id = 0; row_id < chunk_size; ++row_id) {
                kaldi::SubVector<kaldi::BaseFloat> tmp(feature, start);
                kaldi::SubVector<kaldi::BaseFloat> f_chunk_tmp(
                    feature_chunk.Data() + row_id * feature.NumCols(),
                    feature.NumCols());
                f_chunk_tmp.CopyFromVec(tmp);
                ++start;
            }
            raw_data->Accept(feature_chunk);
            if (chunk_idx == num_chunks - 1) {
                raw_data->SetFinished();
            }
            decoder.AdvanceDecode(decodable);
        }
        std::string result;
        result = decoder.GetFinalBestPath();
        decodable->Reset();
        decoder.Reset();
        if (result.empty()) {
            // the TokenWriter can not write empty string.
            ++num_err;
            KALDI_LOG << " the result of " << utt << " is empty";
            continue;
        }
        KALDI_LOG << " the result of " << utt << " is " << result;
        result_writer.Write(utt, result);
        ++num_done;
    }

    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    double elapsed = timer.Elapsed();
    KALDI_LOG << " cost:" << elapsed << " s";
    return (num_done != 0 ? 0 : 1);
}
