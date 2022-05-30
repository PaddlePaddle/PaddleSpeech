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

#include "frontend/audio/feature_pipeline.h"

namespace ppspeech {

using std::unique_ptr;

FeaturePipeline::FeaturePipeline(const FeaturePipelineOptions& opts) {
    unique_ptr<FrontendInterface> data_source(
        new ppspeech::AudioCache(1000 * kint16max, opts.to_float32));

    unique_ptr<FrontendInterface> base_feature;

    if (opts.use_fbank) {
        base_feature.reset(
            new ppspeech::Fbank(opts.fbank_opts, std::move(data_source)));
    } else {
        base_feature.reset(new ppspeech::LinearSpectrogram(
            opts.linear_spectrogram_opts, std::move(data_source)));
    }

    unique_ptr<FrontendInterface> cmvn(
        new ppspeech::CMVN(opts.cmvn_file, std::move(base_feature)));

    unique_ptr<FrontendInterface> cache(
        new ppspeech::FeatureCache(opts.feature_cache_opts, std::move(cmvn)));

    base_extractor_.reset(
        new ppspeech::Dispenser(opts.dispenser_opts, std::move(cache)));
}

}  // ppspeech