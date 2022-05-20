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
#include "kaldi/feat/wave-reader.h"
#include "kaldi/util/kaldi-io.h"
#include "kaldi/util/table-types.h"

#include "frontend/audio/audio_cache.h"
#include "frontend/audio/data_cache.h"
#include "frontend/audio/fbank.h"
#include "frontend/audio/feature_cache.h"
#include "frontend/audio/frontend_itf.h"
#include "frontend/audio/normalizer.h"

DEFINE_string(wav_rspecifier, "", "test wav scp path");
DEFINE_string(feature_wspecifier, "", "output feats wspecifier");
DEFINE_string(cmvn_file, "", "read cmvn");
DEFINE_double(streaming_chunk, 0.36, "streaming feature chunk size");
DEFINE_int32(num_bins, 161, "fbank num bins");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(
        FLAGS_wav_rspecifier);
    kaldi::BaseFloatMatrixWriter feat_writer(FLAGS_feature_wspecifier);

    int32 num_done = 0, num_err = 0;

    // feature pipeline: wave cache --> povey window
    // -->fbank --> global cmvn -> feat cache

    std::unique_ptr<ppspeech::FrontendInterface> data_source(
        new ppspeech::AudioCache(3600 * 1600, false));

    ppspeech::FbankOptions opt;
    opt.fbank_opts.frame_opts.frame_length_ms = 25;
    opt.fbank_opts.frame_opts.frame_shift_ms = 10;
    opt.streaming_chunk = FLAGS_streaming_chunk;
    opt.fbank_opts.mel_opts.num_bins = FLAGS_num_bins;
    opt.fbank_opts.frame_opts.dither = 0.0;

    std::unique_ptr<ppspeech::FrontendInterface> fbank(
        new ppspeech::Fbank(opt, std::move(data_source)));

    std::unique_ptr<ppspeech::FrontendInterface> cmvn(
        new ppspeech::CMVN(FLAGS_cmvn_file, std::move(fbank)));

    ppspeech::FeatureCacheOptions feat_cache_opts;
    // the feature cache output feature chunk by chunk.
    // frame_chunk_size : num frame of a chunk.
    // frame_chunk_stride: chunk sliding window stride.
    feat_cache_opts.frame_chunk_stride = 1;
    feat_cache_opts.frame_chunk_size = 1;
    ppspeech::FeatureCache feature_cache(feat_cache_opts, std::move(cmvn));
    LOG(INFO) << "fbank: " << true;
    LOG(INFO) << "feat dim: " << feature_cache.Dim();

    int sample_rate = 16000;
    float streaming_chunk = FLAGS_streaming_chunk;
    int chunk_sample_size = streaming_chunk * sample_rate;
    LOG(INFO) << "sr: " << sample_rate;
    LOG(INFO) << "chunk size (s): " << streaming_chunk;
    LOG(INFO) << "chunk size (sample): " << chunk_sample_size;

    for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string utt = wav_reader.Key();
        const kaldi::WaveData& wave_data = wav_reader.Value();
        LOG(INFO) << "process utt: " << utt;

        int32 this_channel = 0;
        kaldi::SubVector<kaldi::BaseFloat> waveform(wave_data.Data(),
                                                    this_channel);
        int tot_samples = waveform.Dim();
        LOG(INFO) << "wav len (sample): " << tot_samples;

        int sample_offset = 0;
        std::vector<kaldi::Vector<BaseFloat>> feats;
        int feature_rows = 0;
        while (sample_offset < tot_samples) {
            int cur_chunk_size =
                std::min(chunk_sample_size, tot_samples - sample_offset);

            kaldi::Vector<kaldi::BaseFloat> wav_chunk(cur_chunk_size);
            for (int i = 0; i < cur_chunk_size; ++i) {
                wav_chunk(i) = waveform(sample_offset + i);
            }

            kaldi::Vector<BaseFloat> features;
            feature_cache.Accept(wav_chunk);
            if (cur_chunk_size < chunk_sample_size) {
                feature_cache.SetFinished();
            }
            bool flag = true;
            do {
                flag = feature_cache.Read(&features);
                feats.push_back(features);
                feature_rows += features.Dim() / feature_cache.Dim();
            } while (flag == true && features.Dim() != 0);
            sample_offset += cur_chunk_size;
        }

        int cur_idx = 0;
        kaldi::Matrix<kaldi::BaseFloat> features(feature_rows,
                                                 feature_cache.Dim());
        for (auto feat : feats) {
            int num_rows = feat.Dim() / feature_cache.Dim();
            for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
                for (size_t col_idx = 0; col_idx < feature_cache.Dim();
                     ++col_idx) {
                    features(cur_idx, col_idx) =
                        feat(row_idx * feature_cache.Dim() + col_idx);
                }
                ++cur_idx;
            }
        }
        feat_writer.Write(utt, features);
        feature_cache.Reset();

        if (num_done % 50 == 0 && num_done != 0)
            KALDI_VLOG(2) << "Processed " << num_done << " utterances";
        num_done++;
    }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
