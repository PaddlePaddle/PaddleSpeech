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

#include "nnet/ds2_nnet.h"
#include "base/common.h"
#include "decoder/param.h"
#include "frontend/audio/assembler.h"
#include "frontend/audio/data_cache.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"

DEFINE_string(feature_rspecifier, "", "test feature rspecifier");
DEFINE_string(nnet_prob_wspecifier, "", "nnet porb wspecifier");

using kaldi::BaseFloat;
using kaldi::Matrix;
using std::vector;

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    kaldi::SequentialBaseFloatMatrixReader feature_reader(
        FLAGS_feature_rspecifier);
    kaldi::BaseFloatMatrixWriter nnet_writer(FLAGS_nnet_prob_wspecifier);
    std::string model_graph = FLAGS_model_path;
    std::string model_params = FLAGS_param_path;
    LOG(INFO) << "model path: " << model_graph;
    LOG(INFO) << "model param: " << model_params;

    int32 num_done = 0, num_err = 0;

    ppspeech::ModelOptions model_opts =  ppspeech::ModelOptions::InitFromFlags();

    std::shared_ptr<ppspeech::PaddleNnet> nnet(
        new ppspeech::PaddleNnet(model_opts));
    std::shared_ptr<ppspeech::DataCache> raw_data(new ppspeech::DataCache());
    std::shared_ptr<ppspeech::Decodable> decodable(
        new ppspeech::Decodable(nnet, raw_data, FLAGS_acoustic_scale));

    int32 chunk_size = FLAGS_receptive_field_length +
                       (FLAGS_nnet_decoder_chunk - 1) * FLAGS_subsampling_rate;
    int32 chunk_stride = FLAGS_subsampling_rate * FLAGS_nnet_decoder_chunk;
    int32 receptive_field_length = FLAGS_receptive_field_length;
    LOG(INFO) << "chunk size (frame): " << chunk_size;
    LOG(INFO) << "chunk stride (frame): " << chunk_stride;
    LOG(INFO) << "receptive field (frame): " << receptive_field_length;
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
        int32 frame_idx = 0;
        std::vector<kaldi::Vector<kaldi::BaseFloat>> prob_vec;
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
        decodable->Reset();
        if (prob_vec.size() == 0) {
            // the TokenWriter can not write empty string.
            ++num_err;
            KALDI_LOG << " the nnet prob of " << utt << " is empty";
            continue;
        }
        kaldi::Matrix<kaldi::BaseFloat> result(prob_vec.size(),
                                               prob_vec[0].Dim());
        for (int row_idx = 0; row_idx < prob_vec.size(); ++row_idx) {
            for (int32 col_idx = 0; col_idx < prob_vec[0].Dim(); ++col_idx) {
                result(row_idx, col_idx) = prob_vec[row_idx](col_idx);
            }
        }

        nnet_writer.Write(utt, result);
        ++num_done;
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << " cost:" << elapsed << " s";

    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
