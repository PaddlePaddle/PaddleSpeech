/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>

#include "base/flags.h"
#include "base/log.h"
#include "frontend/audio/audio_cache.h"
#include "frontend/audio/data_cache.h"
#include "frontend/audio/fbank.h"
#include "frontend/audio/feature_cache.h"
#include "frontend/audio/frontend_itf.h"
#include "frontend/audio/normalizer.h"

#include "kaldi-native-fbank/csrc/online-feature.h"

#include "kaldi/feat/wave-reader.h"
#include "kaldi/util/kaldi-io.h"
#include "kaldi/util/table-types.h"
#include <fstream>

int main(int argc, char* argv[]) {
  kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(
        argv[1]);
  kaldi::SequentialTableReader<kaldi::WaveInfoHolder> wav_info_reader(
        argv[1]);

  knf::FbankOptions opts;
  opts.frame_opts.samp_freq = 16000;
  opts.frame_opts.frame_length_ms = 32;
  opts.frame_opts.frame_shift_ms = 10;
  opts.mel_opts.num_bins = 64;
  opts.frame_opts.dither = 0.0;

  for (; !wav_reader.Done() && !wav_info_reader.Done();
         wav_reader.Next(), wav_info_reader.Next()) {
        const std::string& utt = wav_reader.Key();
        const kaldi::WaveData& wave_data = wav_reader.Value();

        const std::string& utt2 = wav_info_reader.Key();
        const kaldi::WaveInfo& wave_info = wav_info_reader.Value();

        CHECK(utt == utt2)
            << "wav reader and wav info reader using diff rspecifier!!!";
        LOG(INFO) << "utt: " << utt;
        LOG(INFO) << "samples: " << wave_info.SampleCount();
        LOG(INFO) << "dur: " << wave_info.Duration() << " sec";
        CHECK(wave_info.SampFreq() == 16000)
            << "need " << 16000 << " get " << wave_info.SampFreq();

        // load first channel wav
        int32 this_channel = 0;
        kaldi::SubVector<kaldi::BaseFloat> waveform(wave_data.Data(),
                                                    this_channel);
        // compute feat chunk by chunk
        int tot_samples = waveform.Dim();
        int sample_offset = 0;
        std::vector<kaldi::Vector<BaseFloat>> feats;
        int feature_rows = 0;

        for (int i = 0; i < tot_samples; i++){
            waveform(i) = waveform(i) / 32768;
        }

        knf::OnlineFbank fbank(opts);
        fbank.AcceptWaveform(16000, waveform.Data(), tot_samples);

        std::ostringstream os;

        int32_t n = fbank.NumFramesReady();
        for (int32_t i = 0; i != n; ++i) {
          const float *frame = fbank.GetFrame(i);
          for (int32_t k = 0; k != opts.mel_opts.num_bins; ++k) {
            // os << frame[k] << ", ";
          }
          // os << "\n";
        }
        // std::cout << os.str() << "\n";
  }
  return 0;
}
