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

#include "websocket/websocket_client.h"
#include "kaldi/feat/wave-reader.h"
#include "kaldi/util/kaldi-io.h"
#include "kaldi/util/table-types.h"

DEFINE_string(host, "127.0.0.1", "host of websocket server");
DEFINE_int32(port, 8082, "port of websocket server");
DEFINE_string(wav_rspecifier, "", "test wav scp path");
DEFINE_double(streaming_chunk, 0.1, "streaming feature chunk size");

using kaldi::int16;
int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(
        FLAGS_wav_rspecifier);

    const int sample_rate = 16000;
    const float streaming_chunk = FLAGS_streaming_chunk;
    const int chunk_sample_size = streaming_chunk * sample_rate;

    for (; !wav_reader.Done(); wav_reader.Next()) {
        ppspeech::WebSocketClient client(FLAGS_host, FLAGS_port);

        client.SendStartSignal();
        std::string utt = wav_reader.Key();
        const kaldi::WaveData& wave_data = wav_reader.Value();
        CHECK_EQ(wave_data.SampFreq(), sample_rate);

        int32 this_channel = 0;
        kaldi::SubVector<kaldi::BaseFloat> waveform(wave_data.Data(),
                                                    this_channel);
        const int tot_samples = waveform.Dim();
        int sample_offset = 0;

        while (sample_offset < tot_samples) {
            int cur_chunk_size =
                std::min(chunk_sample_size, tot_samples - sample_offset);

            std::vector<int16> wav_chunk(cur_chunk_size);
            for (int i = 0; i < cur_chunk_size; ++i) {
                wav_chunk[i] = static_cast<int16>(waveform(sample_offset + i));
            }
            client.SendBinaryData(wav_chunk.data(),
                                  wav_chunk.size() * sizeof(int16));


            sample_offset += cur_chunk_size;
            LOG(INFO) << "Send " << cur_chunk_size << " samples";
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(1 * 1000)));

            if (cur_chunk_size < chunk_sample_size) {
                client.SendEndSignal();
            }
        }

        while (!client.Done()) {
        }
        std::string result = client.GetResult();
        LOG(INFO) << "utt: " << utt << " " << result;

        client.Join();
    }

    return 0;
}
