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

#include "kaldi/feat/wave-reader.h"

#include "websocket/websocket_client.h"

DEFINE_string(host, "127.0.0.1", "host of websocket server");
DEFINE_int32(port, 10086, "port of websocket server");
DEFINE_string(wav_rspecifier, "", "test wav scp path");


using kaldi::int16;
int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    wenet::WebSocketClient client(FLAGS_host, FLAGS_port);

    kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(
        FLAGS_wav_rspecifier);

    const int sample_rate = 16000;
    const float streaming_chunk = 0.36;
    const int chunk_sample_size = streaming_chunk * sample_rate;

    for (; !wav_reader.Done(); wav_reader.Next()) {
        client.SendStartSignal();
        std::string utt = wav_reader.Key();
        const kaldi::WaveData& wave_data = wav_reader.Value();
        CHECK_EQ(wave_data.SampFreq(), sample_rate);
        const int tot_samples = wave_data.Dim();

        while (sample_offset < tot_samples) {
            if (client.done()) {
                break;
            }

            int cur_chunk_size =
                std::min(chunk_sample_size, tot_samples - sample_offset);

            kaldi::Vector<int16> wav_chunk(cur_chunk_size);
            for (int i = 0; i < cur_chunk_size; ++i) {
                wav_chunk(i) = static_cast<int16>(waveform(sample_offset + i));
            }
            client.SendBinaryData(wav_chunk.data(),
                                  wav_chunk.Dim() * sizeof(int16));

            if (cur_chunk_size < chunk_sample_size) {
            }

            sample_offset += cur_chunk_size;
            VLOG(2) << "Send " << data.size() << " samples";
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(streaming_chunk * 1000)));
        }
        client.SendEndSignal();
        client.Join();
        return 0;
    }
    return 0;
}
