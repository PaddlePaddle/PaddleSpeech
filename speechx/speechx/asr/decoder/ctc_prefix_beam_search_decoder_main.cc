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

#include "decoder/ctc_prefix_beam_search_decoder.h"
#include "absl/strings/str_split.h"
#include "base/common.h"
#include "frontend/audio/data_cache.h"
#include "fst/symbol-table.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"
#include "nnet/nnet_producer.h"
#include "nnet/u2_nnet.h"

DEFINE_string(feature_rspecifier, "", "test feature rspecifier");
DEFINE_string(result_wspecifier, "", "test result wspecifier");
DEFINE_string(vocab_path, "", "vocab path");

DEFINE_string(model_path, "", "paddle nnet model");

DEFINE_int32(receptive_field_length,
             7,
             "receptive field of two CNN(kernel=3) downsampling module.");
DEFINE_int32(subsampling_rate,
             4,
             "two CNN(kernel=3) module downsampling rate.");

DEFINE_int32(nnet_decoder_chunk, 16, "paddle nnet forward chunk");

using kaldi::BaseFloat;
using kaldi::Matrix;
using std::vector;

// test u2 online decoder by feeding speech feature
int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    int32 num_done = 0, num_err = 0;

    CHECK_NE(FLAGS_result_wspecifier, "");
    CHECK_NE(FLAGS_feature_rspecifier, "");
    CHECK_NE(FLAGS_vocab_path, "");
    CHECK_NE(FLAGS_model_path, "");
    LOG(INFO) << "model path: " << FLAGS_model_path;
    LOG(INFO) << "Reading vocab table " << FLAGS_vocab_path;

    kaldi::SequentialBaseFloatMatrixReader feature_reader(
        FLAGS_feature_rspecifier);
    kaldi::TokenWriter result_writer(FLAGS_result_wspecifier);

    // nnet
    ppspeech::ModelOptions model_opts;
    model_opts.model_path = FLAGS_model_path;
    std::shared_ptr<ppspeech::U2Nnet> nnet =
        std::make_shared<ppspeech::U2Nnet>(model_opts);

    // decodeable
    std::shared_ptr<ppspeech::DataCache> raw_data =
        std::make_shared<ppspeech::DataCache>();
    std::shared_ptr<ppspeech::NnetProducer> nnet_producer =
        std::make_shared<ppspeech::NnetProducer>(nnet, raw_data);
    std::shared_ptr<ppspeech::Decodable> decodable =
        std::make_shared<ppspeech::Decodable>(nnet_producer);

    // decoder
    ppspeech::CTCBeamSearchOptions opts;
    opts.blank = 0;
    opts.first_beam_size = 10;
    opts.second_beam_size = 10;
    ppspeech::CTCPrefixBeamSearch decoder(FLAGS_vocab_path, opts);


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

        int nframes = feature.NumRows();
        int feat_dim = feature.NumCols();
        raw_data->SetDim(feat_dim);
        LOG(INFO) << "utt: " << utt;
        LOG(INFO) << "feat shape: " << nframes << ", " << feat_dim;

        raw_data->SetDim(feat_dim);

        int32 ori_feature_len = feature.NumRows();
        int32 num_chunks = feature.NumRows() / chunk_stride + 1;
        LOG(INFO) << "num_chunks: " << num_chunks;

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            int32 this_chunk_size = 0;
            if (ori_feature_len > chunk_idx * chunk_stride) {
                this_chunk_size = std::min(
                    ori_feature_len - chunk_idx * chunk_stride, chunk_size);
            }
            if (this_chunk_size < receptive_field_length) {
                LOG(WARNING) << "utt: " << utt << " skip last "
                             << this_chunk_size << " frames, expect is "
                             << receptive_field_length;
                break;
            }


            kaldi::Vector<kaldi::BaseFloat> feature_chunk(this_chunk_size *
                                                          feat_dim);
            int32 start = chunk_idx * chunk_stride;
            for (int row_id = 0; row_id < this_chunk_size; ++row_id) {
                kaldi::SubVector<kaldi::BaseFloat> feat_row(feature, start);
                kaldi::SubVector<kaldi::BaseFloat> feature_chunk_row(
                    feature_chunk.Data() + row_id * feat_dim, feat_dim);

                feature_chunk_row.CopyFromVec(feat_row);
                ++start;
            }

            // feat to frontend pipeline cache
            raw_data->Accept(feature_chunk);

            // send data finish signal
            if (chunk_idx == num_chunks - 1) {
                raw_data->SetFinished();
            }

            // forward nnet
            decoder.AdvanceDecode(decodable);

            LOG(INFO) << "Partial result: " << decoder.GetPartialResult();
        }

        decoder.FinalizeSearch();

        // get 1-best result
        std::string result = decoder.GetFinalBestPath();

        // after process one utt, then reset state.
        decodable->Reset();
        decoder.Reset();

        if (result.empty()) {
            // the TokenWriter can not write empty string.
            ++num_err;
            LOG(INFO) << " the result of " << utt << " is empty";
            continue;
        }

        LOG(INFO) << " the result of " << utt << " is " << result;
        result_writer.Write(utt, result);

        ++num_done;
    }

    double elapsed = timer.Elapsed();
    LOG(INFO) << "Program cost:" << elapsed << " sec";

    LOG(INFO) << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
