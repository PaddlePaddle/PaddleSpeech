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

#ifndef USE_ONNX
    #include "nnet/u2_nnet.h"
#else
    #include "nnet/u2_onnx_nnet.h"
#endif
#include "base/common.h"
#include "decoder/param.h"
#include "frontend/feature_pipeline.h"
#include "frontend/wave-reader.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"
#include "nnet/nnet_producer.h"
#include "nnet/u2_nnet.h"

DEFINE_string(wav_rspecifier, "", "test wav rspecifier");
DEFINE_string(nnet_prob_wspecifier, "", "nnet porb wspecifier");
DEFINE_double(streaming_chunk, 0.36, "streaming feature chunk size");
DEFINE_int32(sample_rate, 16000, "sample rate");

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
    int sample_rate = FLAGS_sample_rate;
    float streaming_chunk = FLAGS_streaming_chunk;
    int chunk_sample_size = streaming_chunk * sample_rate;

    CHECK_GT(FLAGS_wav_rspecifier.size(), 0);
    CHECK_GT(FLAGS_nnet_prob_wspecifier.size(), 0);
    CHECK_GT(FLAGS_model_path.size(), 0);
    LOG(INFO) << "input rspecifier: " << FLAGS_wav_rspecifier;
    LOG(INFO) << "output wspecifier: " << FLAGS_nnet_prob_wspecifier;
    LOG(INFO) << "model path: " << FLAGS_model_path;

    kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(
        FLAGS_wav_rspecifier);
    kaldi::BaseFloatMatrixWriter nnet_out_writer(FLAGS_nnet_prob_wspecifier);

    ppspeech::ModelOptions model_opts = ppspeech::ModelOptions::InitFromFlags();
    ppspeech::FeaturePipelineOptions feature_opts =
        ppspeech::FeaturePipelineOptions::InitFromFlags();
    feature_opts.assembler_opts.fill_zero = false;

#ifndef USE_ONNX
    std::shared_ptr<ppspeech::U2Nnet> nnet(new ppspeech::U2Nnet(model_opts));
#else
    std::shared_ptr<ppspeech::U2OnnxNnet> nnet(new ppspeech::U2OnnxNnet(model_opts));
#endif
    std::shared_ptr<ppspeech::FeaturePipeline> feature_pipeline(
        new ppspeech::FeaturePipeline(feature_opts));
    std::shared_ptr<ppspeech::NnetProducer> nnet_producer(
        new ppspeech::NnetProducer(nnet, feature_pipeline));
    kaldi::Timer timer;
    float tot_wav_duration = 0;

    for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string utt = wav_reader.Key();
        const kaldi::WaveData& wave_data = wav_reader.Value();
        LOG(INFO) << "utt: " << utt;
        LOG(INFO) << "wav dur: " << wave_data.Duration() << " sec.";
        double dur = wave_data.Duration();
        tot_wav_duration += dur;

        int32 this_channel = 0;
        kaldi::SubVector<kaldi::BaseFloat> waveform(wave_data.Data(),
                                                    this_channel);
        int tot_samples = waveform.Dim();
        LOG(INFO) << "wav len (sample): " << tot_samples;

        int sample_offset = 0;
        kaldi::Timer timer;

        while (sample_offset < tot_samples) {
            int cur_chunk_size =
                std::min(chunk_sample_size, tot_samples - sample_offset);

            std::vector<kaldi::BaseFloat> wav_chunk(cur_chunk_size);
            for (int i = 0; i < cur_chunk_size; ++i) {
                wav_chunk[i] = waveform(sample_offset + i);
            }

            nnet_producer->Accept(wav_chunk);
            if (cur_chunk_size < chunk_sample_size) {
                nnet_producer->SetInputFinished();
            }

            // no overlap
            sample_offset += cur_chunk_size;
        }
        CHECK(sample_offset == tot_samples);

        std::vector<std::vector<kaldi::BaseFloat>> prob_vec;
        while (1) {
            std::vector<kaldi::BaseFloat> logprobs;
            bool isok = nnet_producer->Read(&logprobs);
            if (nnet_producer->IsFinished()) break;
            if (isok == false) continue;
            prob_vec.push_back(logprobs);
        }
        {
            // writer nnet output
            kaldi::MatrixIndexT nrow = prob_vec.size();
            kaldi::MatrixIndexT ncol = prob_vec[0].size();
            LOG(INFO) << "nnet out shape: " << nrow << ", " << ncol;
            kaldi::Matrix<kaldi::BaseFloat> nnet_out(nrow, ncol);
            for (int32 row_idx = 0; row_idx < nrow; ++row_idx) {
                for (int32 col_idx = 0; col_idx < ncol; ++col_idx) {
                    nnet_out(row_idx, col_idx) = prob_vec[row_idx][col_idx];
                }
            }
            nnet_out_writer.Write(utt, nnet_out);
        }
        nnet_producer->Reset();
    }

    nnet_producer->Wait();
    double elapsed = timer.Elapsed();
    LOG(INFO) << "Program cost:" << elapsed << " sec";

    LOG(INFO) << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
