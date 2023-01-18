// Copyright (c) 2016 Personal (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef FRONTEND_WAV_H_
#define FRONTEND_WAV_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#pragma once

namespace ppspeech {

struct WavHeader {
  char riff[4] = {'R', 'I', 'F', 'F'};
  unsigned int size = 0;
  char wav[4] = {'W', 'A', 'V', 'E'};
  char fmt[4] = {'f', 'm', 't', ' '};
  unsigned int fmt_size = 16;
  uint16_t format = 1;
  uint16_t channels = 0;
  unsigned int sample_rate = 0;
  unsigned int bytes_per_second = 0;
  uint16_t block_size = 0;
  uint16_t bit = 0;
  char data[4] = {'d', 'a', 't', 'a'};
  unsigned int data_size = 0;

  WavHeader() {}

  WavHeader(int num_samples, int num_channel, int sample_rate,
            int bits_per_sample) {
    data_size = num_samples * num_channel * (bits_per_sample / 8);
    size = sizeof(WavHeader) - 8 + data_size;
    channels = num_channel;
    this->sample_rate = sample_rate;
    bytes_per_second = sample_rate * num_channel * (bits_per_sample / 8);
    block_size = num_channel * (bits_per_sample / 8);
    bit = bits_per_sample;
  }
};

class WavReader {
 public:
  WavReader() : data_(nullptr) {}
  explicit WavReader(const std::string& filename) { Open(filename); }

  bool Open(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (NULL == fp) {
      //LOG(WARNING) << "Error in read " << filename;
      return false;
    }

    WavHeader header;
    fread(&header, 1, sizeof(header), fp);
    if (header.fmt_size < 16) {
      fprintf(stderr,
              "WaveData: expect PCM format data "
              "to have fmt chunk of at least size 16.\n");
      return false;
    } else if (header.fmt_size > 16) {
      int offset = 44 - 8 + header.fmt_size - 16;
      fseek(fp, offset, SEEK_SET);
      fread(header.data, 8, sizeof(char), fp);
    }
    // check "RIFF" "WAVE" "fmt " "data"

    // Skip any sub-chunks between "fmt" and "data".  Usually there will
    // be a single "fact" sub chunk, but on Windows there can also be a
    // "list" sub chunk.
    while (0 != strncmp(header.data, "data", 4)) {
      // We will just ignore the data in these chunks.
      fseek(fp, header.data_size, SEEK_CUR);
      // read next sub chunk
      fread(header.data, 8, sizeof(char), fp);
    }

    num_channel_ = header.channels;
    sample_rate_ = header.sample_rate;
    bits_per_sample_ = header.bit;
    int num_data = header.data_size / (bits_per_sample_ / 8);
    data_ = new float[num_data];
    num_samples_ = num_data / num_channel_;

    for (int i = 0; i < num_data; ++i) {
      switch (bits_per_sample_) {
        case 8: {
          char sample;
          fread(&sample, 1, sizeof(char), fp);
          data_[i] = static_cast<float>(sample);
          break;
        }
        case 16: {
          int16_t sample;
          fread(&sample, 1, sizeof(int16_t), fp);
          data_[i] = static_cast<float>(sample);
          break;
        }
        case 32: {
          int sample;
          fread(&sample, 1, sizeof(int), fp);
          data_[i] = static_cast<float>(sample);
          break;
        }
        default:
          fprintf(stderr, "unsupported quantization bits");
          exit(1);
      }
    }
    fclose(fp);
    return true;
  }

  int num_channel() const { return num_channel_; }
  int sample_rate() const { return sample_rate_; }
  int bits_per_sample() const { return bits_per_sample_; }
  int num_samples() const { return num_samples_; }

  ~WavReader() {
    delete[] data_;
  }

  const float* data() const { return data_; }

 private:
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
  int num_samples_;  // sample points per channel
  float* data_;
};

class WavWriter {
 public:
  WavWriter(const float* data, int num_samples, int num_channel,
            int sample_rate, int bits_per_sample)
      : data_(data),
        num_samples_(num_samples),
        num_channel_(num_channel),
        sample_rate_(sample_rate),
        bits_per_sample_(bits_per_sample) {}

  void Write(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    WavHeader header(num_samples_, num_channel_, sample_rate_,
                     bits_per_sample_);
    fwrite(&header, 1, sizeof(header), fp);

    for (int i = 0; i < num_samples_; ++i) {
      for (int j = 0; j < num_channel_; ++j) {
        switch (bits_per_sample_) {
          case 8: {
            char sample = static_cast<char>(data_[i * num_channel_ + j]);
            fwrite(&sample, 1, sizeof(sample), fp);
            break;
          }
          case 16: {
            int16_t sample = static_cast<int16_t>(data_[i * num_channel_ + j]);
            fwrite(&sample, 1, sizeof(sample), fp);
            break;
          }
          case 32: {
            int sample = static_cast<int>(data_[i * num_channel_ + j]);
            fwrite(&sample, 1, sizeof(sample), fp);
            break;
          }
        }
      }
    }
    fclose(fp);
  }

 private:
  const float* data_;
  int num_samples_;  // total float points in data_
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
};

class StreamWavWriter {
 public:
  StreamWavWriter(int num_channel, int sample_rate, int bits_per_sample)
     : num_channel_(num_channel),
       sample_rate_(sample_rate),
       bits_per_sample_(bits_per_sample),
       total_num_samples_(0) {}

  StreamWavWriter(const std::string& filename, int num_channel,
                  int sample_rate, int bits_per_sample)
     : StreamWavWriter(num_channel, sample_rate, bits_per_sample) {
    Open(filename);
  }

  void Open(const std::string& filename) {
    fp_ = fopen(filename.c_str(), "wb");
    fseek(fp_, sizeof(WavHeader), SEEK_SET);
  }

  void Write(const int16_t* sample_data, size_t num_samples) {
    fwrite(sample_data, sizeof(int16_t), num_samples, fp_);
    total_num_samples_ += num_samples;
  }

  void Close() {
    WavHeader header(total_num_samples_, num_channel_, sample_rate_,
                     bits_per_sample_);
    fseek(fp_, 0L, SEEK_SET);
    fwrite(&header, 1, sizeof(header), fp_);
    fclose(fp_);
  }

 private:
  FILE* fp_;
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
  size_t total_num_samples_;
};

}  // namespace wenet

#endif  // FRONTEND_WAV_H_
