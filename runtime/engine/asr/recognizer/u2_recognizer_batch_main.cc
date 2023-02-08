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

#include "recognizer/u2_recognizer.h"
#include "common/base/thread_pool.h"
#include "common/utils/file_utils.h"
#include "common/utils/strings.h"
#include "decoder/param.h"
#include "frontend/wave-reader.h"
#include "kaldi/util/table-types.h"
#include "nnet/u2_nnet.h"

DEFINE_string(wav_rspecifier, "", "test feature rspecifier");
DEFINE_string(result_wspecifier, "", "test result wspecifier");
DEFINE_double(streaming_chunk, 0.36, "streaming feature chunk size");
DEFINE_int32(sample_rate, 16000, "sample rate");
DEFINE_int32(njob, 3, "njob");

using std::string;
using std::vector;

void SplitUtt(string wavlist_file,
              vector<vector<string>>* uttlists,
              vector<vector<string>>* wavlists,
              int njob) {
    vector<string> wavlist;
    wavlists->resize(njob);
    uttlists->resize(njob);
    ppspeech::ReadFileToVector(wavlist_file, &wavlist);
    for (size_t idx = 0; idx < wavlist.size(); ++idx) {
        string utt_str = wavlist[idx];
        vector<string> utt_wav = ppspeech::StrSplit(utt_str, " \t");
        LOG(INFO) << utt_wav[0];
        CHECK_EQ(utt_wav.size(), size_t(2));
        uttlists->at(idx % njob).push_back(utt_wav[0]);
        wavlists->at(idx % njob).push_back(utt_wav[1]);
    }
}

void recognizer_func(const ppspeech::U2RecognizerResource& resource,
                     std::shared_ptr<ppspeech::NnetBase> nnet,
                     std::vector<string> wavlist,
                     std::vector<string> uttlist,
                     std::vector<string>* results) {
    int32 num_done = 0, num_err = 0;
    double tot_wav_duration = 0.0;
    double tot_attention_rescore_time = 0.0;
    double tot_decode_time = 0.0;
    int chunk_sample_size = FLAGS_streaming_chunk * FLAGS_sample_rate;
    if (wavlist.empty()) return;

    std::shared_ptr<ppspeech::U2Recognizer> recognizer_ptr =
        std::make_shared<ppspeech::U2Recognizer>(resource, nnet);

    results->reserve(wavlist.size());
    for (size_t idx = 0; idx < wavlist.size(); ++idx) {
        std::string utt = uttlist[idx];
        std::string wav_file = wavlist[idx];
        std::ifstream infile;
        infile.open(wav_file, std::ifstream::in);
        kaldi::WaveData wave_data;
        wave_data.Read(infile);
        recognizer_ptr->InitDecoder();
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
        kaldi::Timer local_timer;

        while (sample_offset < tot_samples) {
            int cur_chunk_size =
                std::min(chunk_sample_size, tot_samples - sample_offset);

            std::vector<kaldi::BaseFloat> wav_chunk(cur_chunk_size);
            for (int i = 0; i < cur_chunk_size; ++i) {
                wav_chunk[i] = waveform(sample_offset + i);
            }

            recognizer_ptr->Accept(wav_chunk);
            if (cur_chunk_size < chunk_sample_size) {
                recognizer_ptr->SetInputFinished();
            }

            // no overlap
            sample_offset += cur_chunk_size;
        }
        CHECK(sample_offset == tot_samples);
        recognizer_ptr->WaitDecodeFinished();

        kaldi::Timer timer;
        recognizer_ptr->AttentionRescoring();
        tot_attention_rescore_time += timer.Elapsed();

        std::string result = recognizer_ptr->GetFinalResult();
        if (result.empty()) {
            // the TokenWriter can not write empty string.
            ++num_err;
            LOG(INFO) << " the result of " << utt << " is empty";
            result = " ";
        }

        tot_decode_time += local_timer.Elapsed();
        LOG(INFO) << utt << " " << result;
        LOG(INFO) << " RTF: " << local_timer.Elapsed() / dur << " dur: " << dur
                  << " cost: " << local_timer.Elapsed();

        results->push_back(result);
        ++num_done;
    }
    recognizer_ptr->WaitFinished();
    LOG(INFO) << "Done " << num_done << " out of " << (num_err + num_done);
    LOG(INFO) << "total wav duration is: " << tot_wav_duration << " sec";
    LOG(INFO) << "total decode cost:" << tot_decode_time << " sec";
    LOG(INFO) << "total rescore cost:" << tot_attention_rescore_time << " sec";
    LOG(INFO) << "RTF is: " << tot_decode_time / tot_wav_duration;
}

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    int sample_rate = FLAGS_sample_rate;
    float streaming_chunk = FLAGS_streaming_chunk;
    int chunk_sample_size = streaming_chunk * sample_rate;
    kaldi::TokenWriter result_writer(FLAGS_result_wspecifier);
    int njob = FLAGS_njob;
    LOG(INFO) << "sr: " << sample_rate;
    LOG(INFO) << "chunk size (s): " << streaming_chunk;
    LOG(INFO) << "chunk size (sample): " << chunk_sample_size;

    ppspeech::U2RecognizerResource resource =
        ppspeech::U2RecognizerResource::InitFromFlags();
    ThreadPool threadpool(njob);
    vector<vector<string>> wavlist;
    vector<vector<string>> uttlist;
    vector<vector<string>> resultlist(njob);
    vector<std::future<void>> futurelist;
    std::shared_ptr<ppspeech::U2Nnet> nnet(
        new ppspeech::U2Nnet(resource.model_opts));
    SplitUtt(FLAGS_wav_rspecifier, &uttlist, &wavlist, njob);
    for (size_t i = 0; i < njob; ++i) {
        std::future<void> f = threadpool.enqueue(recognizer_func,
                                                 resource,
                                                 nnet->Clone(),
                                                 wavlist[i],
                                                 uttlist[i],
                                                 &resultlist[i]);
        futurelist.push_back(std::move(f));
    }

    for (size_t i = 0; i < njob; ++i) {
        futurelist[i].get();
    }

    for (size_t idx = 0; idx < njob; ++idx) {
        for (size_t utt_idx = 0; utt_idx < uttlist[idx].size(); ++utt_idx) {
            string utt = uttlist[idx][utt_idx];
            string result = resultlist[idx][utt_idx];
            result_writer.Write(utt, result);
        }
    }
    return 0;
}
