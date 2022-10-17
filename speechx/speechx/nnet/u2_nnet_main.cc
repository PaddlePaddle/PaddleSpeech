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
#include "frontend/audio/assembler.h"
#include "frontend/audio/data_cache.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"
#include "decoder/param.h"
#include "nnet/u2_nnet.h"


DEFINE_string(feature_rspecifier, "", "test feature rspecifier");
DEFINE_string(nnet_prob_wspecifier, "", "nnet porb wspecifier");
DEFINE_string(nnet_encoder_outs_wspecifier, "", "nnet encoder outs wspecifier");

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

    kaldi::SequentialBaseFloatMatrixReader feature_reader(FLAGS_feature_rspecifier);
    kaldi::BaseFloatMatrixWriter nnet_out_writer(FLAGS_nnet_prob_wspecifier);
    kaldi::BaseFloatMatrixWriter nnet_encoder_outs_writer(FLAGS_nnet_encoder_outs_wspecifier);

    ppspeech::ModelOptions model_opts = ppspeech::ModelOptions::InitFromFlags();

    int32 chunk_size =
        (FLAGS_nnet_decoder_chunk - 1) * FLAGS_subsampling_rate +
        FLAGS_receptive_field_length;
    int32 chunk_stride = FLAGS_subsampling_rate * FLAGS_nnet_decoder_chunk;
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

        int32 frame_idx = 0;
        int vocab_dim = 0;
        std::vector<kaldi::Vector<kaldi::BaseFloat>> prob_vec;
        std::vector<kaldi::Vector<kaldi::BaseFloat>> encoder_out_vec;
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

            // get nnet outputs
            kaldi::Timer timer;
            kaldi::Vector<kaldi::BaseFloat> logprobs;
            bool isok = decodable->AdvanceChunk(&logprobs, &vocab_dim);
            CHECK(isok == true);
            for (int row_idx = 0; row_idx < logprobs.Dim() / vocab_dim; row_idx ++) {
                kaldi::Vector<kaldi::BaseFloat> vec_tmp(vocab_dim);
                std::memcpy(vec_tmp.Data(), logprobs.Data() + row_idx*vocab_dim, sizeof(kaldi::BaseFloat) * vocab_dim);
                prob_vec.push_back(vec_tmp);
            }

            VLOG(2) << "frame_idx: " << frame_idx << " elapsed: " << timer.Elapsed() << " sec.";
        }

        // get encoder out
        decodable->Nnet()->EncoderOuts(&encoder_out_vec);

        // after process one utt, then reset decoder state.
        decodable->Reset();

        if (prob_vec.size() == 0 || encoder_out_vec.size() == 0) {
            // the TokenWriter can not write empty string.
            ++num_err;
            LOG(WARNING) << " the nnet prob/encoder_out of " << utt << " is empty";
            continue;
        }

        {
            // writer nnet output
            kaldi::MatrixIndexT nrow = prob_vec.size();
            kaldi::MatrixIndexT ncol = prob_vec[0].Dim();
            LOG(INFO) << "nnet out shape: " << nrow << ", " << ncol;
            kaldi::Matrix<kaldi::BaseFloat> nnet_out(nrow, ncol);
            for (int32 row_idx = 0; row_idx < nrow; ++row_idx) {
                for (int32 col_idx = 0; col_idx < ncol; ++col_idx) {
                    nnet_out(row_idx, col_idx) = prob_vec[row_idx](col_idx);
                }
            }
            nnet_out_writer.Write(utt, nnet_out);
        }


        {
            // writer nnet encoder outs
            kaldi::MatrixIndexT nrow = encoder_out_vec.size();
            kaldi::MatrixIndexT ncol = encoder_out_vec[0].Dim();
            LOG(INFO) << "nnet encoder outs shape: " << nrow << ", " << ncol;
            kaldi::Matrix<kaldi::BaseFloat> encoder_outs(nrow, ncol);
            for (int32 row_idx = 0; row_idx < nrow; ++row_idx) {
                for (int32 col_idx = 0; col_idx < ncol; ++col_idx) {
                    encoder_outs(row_idx, col_idx) = encoder_out_vec[row_idx](col_idx);
                }
            }
            nnet_encoder_outs_writer.Write(utt, encoder_outs);
        }

        ++num_done;
    }


    double elapsed = timer.Elapsed();
    LOG(INFO) << "Program cost:" << elapsed << " sec";

    LOG(INFO) << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
