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

#include "nnet/u2_nnet.h"
#include "base/common.h"
#include "frontend/audio/assembler.h"
#include "frontend/audio/data_cache.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"

DEFINE_string(feature_rspecifier, "", "test feature rspecifier");
DEFINE_string(nnet_prob_wspecifier, "", "nnet porb wspecifier");

DEFINE_string(model_path, "", "paddle nnet model");

DEFINE_int32(nnet_decoder_chunk, 16, "nnet forward chunk");
DEFINE_int32(receptive_field_length,
             7,
             "receptive field of two CNN(kernel=3) downsampling module.");
DEFINE_int32(downsampling_rate,
             4,
             "two CNN(kernel=3) module downsampling rate.");
DEFINE_double(acoustic_scale, 1.0, "acoustic scale");

using kaldi::BaseFloat;
using kaldi::Matrix;
using std::vector;

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    int32 num_done = 0, num_err = 0;

    CHECK(FLAGS_feature_rspecifier.size() > 0);
    CHECK(FLAGS_nnet_prob_wspecifier.size() > 0);
    CHECK(FLAGS_model_path.size() > 0);
    LOG(INFO) << "input rspecifier: " << FLAGS_feature_rspecifier;
    LOG(INFO) << "output wspecifier: " << FLAGS_nnet_prob_wspecifier;
    LOG(INFO) << "model path: " << FLAGS_model_path;
    kaldi::SequentialBaseFloatMatrixReader feature_reader(
        FLAGS_feature_rspecifier);
    kaldi::BaseFloatMatrixWriter nnet_out_writer(FLAGS_nnet_prob_wspecifier);

    ppspeech::U2ModelOptions model_opts;
    model_opts.model_path = FLAGS_model_path;

    int32 chunk_size =
        (FLAGS_nnet_decoder_chunk - 1) * FLAGS_downsampling_rate +
        FLAGS_receptive_field_length;
    int32 chunk_stride = FLAGS_downsampling_rate * FLAGS_nnet_decoder_chunk;
    int32 receptive_field_length = FLAGS_receptive_field_length;
    LOG(INFO) << "chunk size (frame): " << chunk_size;
    LOG(INFO) << "chunk stride (frame): " << chunk_stride;
    LOG(INFO) << "receptive field (frame): " << receptive_field_length;

    std::shared_ptr<ppspeech::U2Nnet> nnet(new ppspeech::U2Nnet(model_opts));
    std::shared_ptr<ppspeech::DataCache> raw_data(new ppspeech::DataCache());
    std::shared_ptr<ppspeech::Decodable> decodable(
        new ppspeech::Decodable(nnet, raw_data, FLAGS_acoustic_scale));
    kaldi::Timer timer;

    for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        kaldi::Matrix<BaseFloat> feature = feature_reader.Value();

        int nframes = feature.NumRows();
        int feat_dim = feature.NumCols();
        raw_data->SetDim(feat_dim);
        LOG(INFO) << "utt: " << utt;
        LOG(INFO) << "feat shape: " << nframes << ", " << feat_dim;

        // // pad feats
        // int32 padding_len = 0;
        // if ((feature.NumRows() - chunk_size) % chunk_stride != 0) {
        //     padding_len =
        //         chunk_stride - (feature.NumRows() - chunk_size) %
        //         chunk_stride;
        //     feature.Resize(feature.NumRows() + padding_len,
        //                    feature.NumCols(),
        //                    kaldi::kCopyData);
        // }

        int32 num_chunks = (feature.NumRows() - chunk_size) / chunk_stride + 1;
        int32 frame_idx = 0;
        std::vector<kaldi::Vector<kaldi::BaseFloat>> prob_vec;
        int32 ori_feature_len = feature.NumRows();

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            kaldi::Vector<kaldi::BaseFloat> feature_chunk(chunk_size *
                                                          feat_dim);

            int32 feature_chunk_size = 0;
            if (ori_feature_len > chunk_idx * chunk_stride) {
                feature_chunk_size = std::min(
                    ori_feature_len - chunk_idx * chunk_stride, chunk_size);
            }
            if (feature_chunk_size < receptive_field_length) {
                LOG(WARNING) << "utt: " << utt << " skip last "
                             << feature_chunk_size << " frames, expect is "
                             << receptive_field_length;
                break;
            }

            int32 start = chunk_idx * chunk_stride;
            for (int row_id = 0; row_id < chunk_size; ++row_id) {
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

            // get nnet outputs
            vector<kaldi::BaseFloat> prob;
            while (decodable->FrameLikelihood(frame_idx, &prob)) {
                kaldi::Vector<kaldi::BaseFloat> vec_tmp(prob.size());
                std::memcpy(vec_tmp.Data(),
                            prob.data(),
                            sizeof(kaldi::BaseFloat) * prob.size());
                prob_vec.push_back(vec_tmp);
                frame_idx++;
            }
        }

        // after process one utt, then reset decoder state.
        decodable->Reset();

        if (prob_vec.size() == 0) {
            // the TokenWriter can not write empty string.
            ++num_err;
            LOG(WARNING) << " the nnet prob of " << utt << " is empty";
            continue;
        }

        // writer nnet output
        kaldi::MatrixIndexT nrow = prob_vec.size();
        kaldi::MatrixIndexT ncol = prob_vec[0].Dim();
        LOG(INFO) << "nnet out shape: " << nrow << ", " << ncol;
        kaldi::Matrix<kaldi::BaseFloat> result(nrow, ncol);
        for (int32 row_idx = 0; row_idx < nrow; ++row_idx) {
            for (int32 col_idx = 0; col_idx < ncol; ++col_idx) {
                result(row_idx, col_idx) = prob_vec[row_idx](col_idx);
            }
        }
        nnet_out_writer.Write(utt, result);

        ++num_done;
    }

    double elapsed = timer.Elapsed();
    LOG(INFO) << " cost:" << elapsed << " sec";

    LOG(INFO) << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
