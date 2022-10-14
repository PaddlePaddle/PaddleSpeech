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

#include "decoder/u2_recognizer.h"
#include "decoder/param.h"
#include "kaldi/feat/wave-reader.h"
#include "kaldi/util/table-types.h"

DEFINE_string(wav_rspecifier, "", "test feature rspecifier");
DEFINE_string(result_wspecifier, "", "test result wspecifier");
DEFINE_double(streaming_chunk, 0.36, "streaming feature chunk size");
DEFINE_int32(sample_rate, 16000, "sample rate");


ppspeech::U2RecognizerResource InitOpts() {
    ppspeech::U2RecognizerResource resource;
    resource.acoustic_scale = FLAGS_acoustic_scale;
    resource.feature_pipeline_opts = ppspeech::InitFeaturePipelineOptions();

    ppspeech::ModelOptions model_opts;
    model_opts.model_path = FLAGS_model_path;

    resource.model_opts = model_opts;

    ppspeech::DecodeOptions decoder_opts;
    decoder_opts.chunk_size=16;
    decoder_opts.num_left_chunks = -1;
    decoder_opts.ctc_weight = 0.5;
    decoder_opts.rescoring_weight = 1.0;
    decoder_opts.reverse_weight = 0.3;
    decoder_opts.ctc_prefix_search_opts.blank = 0;
    decoder_opts.ctc_prefix_search_opts.first_beam_size = 10;
    decoder_opts.ctc_prefix_search_opts.second_beam_size = 10;

    resource.decoder_opts = decoder_opts;
    return resource;
}

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    int32 num_done = 0, num_err = 0;
    double tot_wav_duration = 0.0;

    ppspeech::U2RecognizerResource resource = InitOpts();
    ppspeech::U2Recognizer recognizer(resource);

    kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(
        FLAGS_wav_rspecifier);
    kaldi::TokenWriter result_writer(FLAGS_result_wspecifier);

    int sample_rate = FLAGS_sample_rate;
    float streaming_chunk = FLAGS_streaming_chunk;
    int chunk_sample_size = streaming_chunk * sample_rate;
    LOG(INFO) << "sr: " << sample_rate;
    LOG(INFO) << "chunk size (s): " << streaming_chunk;
    LOG(INFO) << "chunk size (sample): " << chunk_sample_size;

    kaldi::Timer timer;

    for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string utt = wav_reader.Key();
        const kaldi::WaveData& wave_data = wav_reader.Value();
        LOG(INFO) << "utt: " << utt;
        LOG(INFO) << "wav dur: " << wave_data.Duration() << " sec.";
        tot_wav_duration += wave_data.Duration();

        int32 this_channel = 0;
        kaldi::SubVector<kaldi::BaseFloat> waveform(wave_data.Data(),
                                                    this_channel);
        int tot_samples = waveform.Dim();
        LOG(INFO) << "wav len (sample): " << tot_samples;

        int sample_offset = 0;
        while (sample_offset < tot_samples) {
            int cur_chunk_size =
                std::min(chunk_sample_size, tot_samples - sample_offset);

            kaldi::Vector<kaldi::BaseFloat> wav_chunk(cur_chunk_size);
            for (int i = 0; i < cur_chunk_size; ++i) {
                wav_chunk(i) = waveform(sample_offset + i);
            }
            // wav_chunk = waveform.Range(sample_offset + i, cur_chunk_size);

            recognizer.Accept(wav_chunk);
            if (cur_chunk_size < chunk_sample_size) {
                recognizer.SetFinished();
            }
            recognizer.Decode();
            LOG(INFO) << "Pratial result: " << recognizer.GetPartialResult();

            // no overlap
            sample_offset += cur_chunk_size;
        }
        // second pass decoding
        recognizer.Rescoring();

        std::string result = recognizer.GetFinalResult();

        recognizer.Reset();

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
  
    LOG(INFO) << "Done " << num_done << " out of " << (num_err + num_done);
    LOG(INFO) << "cost:" << elapsed << " sec";
    LOG(INFO) << "total wav duration is: " << tot_wav_duration << " sec";
    LOG(INFO) << "the RTF is: " << elapsed / tot_wav_duration;
}
