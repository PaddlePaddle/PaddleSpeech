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

// todo refactor later (SGoat)

#pragma once

#include "frontend/assembler.h"
#include "frontend/audio_cache.h"
#include "frontend/cmvn.h"
#include "frontend/data_cache.h"
#include "frontend/fbank.h"
#include "frontend/feature_cache.h"
#include "frontend/frontend_itf.h"

// feature
DECLARE_bool(fill_zero);
DECLARE_int32(num_bins);
DECLARE_string(cmvn_file);

// feature sliding window
DECLARE_int32(receptive_field_length);
DECLARE_int32(subsampling_rate);
DECLARE_int32(nnet_decoder_chunk);

namespace ppspeech {

struct FeaturePipelineOptions {
    std::string cmvn_file{};
    knf::FbankOptions fbank_opts{};
    AssemblerOptions assembler_opts{};

    static FeaturePipelineOptions InitFromFlags() {
        FeaturePipelineOptions opts;
        opts.cmvn_file = FLAGS_cmvn_file;
        LOG(INFO) << "cmvn file: " << opts.cmvn_file;

        // frame options
        knf::FrameExtractionOptions frame_opts;
        frame_opts.dither = 0.0;
        LOG(INFO) << "dither: " << frame_opts.dither;
        frame_opts.frame_shift_ms = 10;
        LOG(INFO) << "frame shift ms: " << frame_opts.frame_shift_ms;
        frame_opts.window_type = "povey";
        frame_opts.frame_length_ms = 25;
        opts.fbank_opts.mel_opts.num_bins = FLAGS_num_bins;
        LOG(INFO) << "num bins: " << opts.fbank_opts.mel_opts.num_bins;

        opts.fbank_opts.frame_opts = frame_opts;
        LOG(INFO) << "frame length ms: " << frame_opts.frame_length_ms;

        // assembler opts
        opts.assembler_opts.subsampling_rate = FLAGS_subsampling_rate;
        opts.assembler_opts.receptive_filed_length =
            FLAGS_receptive_field_length;
        opts.assembler_opts.nnet_decoder_chunk = FLAGS_nnet_decoder_chunk;
        opts.assembler_opts.fill_zero = FLAGS_fill_zero;
        LOG(INFO) << "subsampling rate: "
                  << opts.assembler_opts.subsampling_rate;
        LOG(INFO) << "nnet receptive filed length: "
                  << opts.assembler_opts.receptive_filed_length;
        LOG(INFO) << "nnet chunk size: "
                  << opts.assembler_opts.nnet_decoder_chunk;
        LOG(INFO) << "frontend fill zeros: " << opts.assembler_opts.fill_zero;
        return opts;
    }
};


class FeaturePipeline : public FrontendInterface {
  public:
    explicit FeaturePipeline(const FeaturePipelineOptions& opts);
    virtual void Accept(const std::vector<kaldi::BaseFloat>& waves) {
        base_extractor_->Accept(waves);
    }
    virtual bool Read(std::vector<kaldi::BaseFloat>* feats) {
        return base_extractor_->Read(feats);
    }
    virtual size_t Dim() const { return base_extractor_->Dim(); }
    virtual void SetFinished() { base_extractor_->SetFinished(); }
    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }
    virtual void Reset() { base_extractor_->Reset(); }

    const FeaturePipelineOptions& Config() { return opts_; }

    const BaseFloat FrameShift() const {
        return opts_.fbank_opts.frame_opts.frame_shift_ms;
    }
    const BaseFloat FrameLength() const {
        return opts_.fbank_opts.frame_opts.frame_length_ms;
    }
    const BaseFloat SampleRate() const {
        return opts_.fbank_opts.frame_opts.samp_freq;
    }

  private:
    FeaturePipelineOptions opts_;
    std::unique_ptr<FrontendInterface> base_extractor_;
};

}  // namespace ppspeech
