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

#include "frontend/feature_pipeline.h"

namespace ppspeech {

using std::unique_ptr;

FeaturePipeline::FeaturePipeline(const FeaturePipelineOptions& opts)
    : opts_(opts) {
    unique_ptr<FrontendInterface> data_source(
        new ppspeech::AudioCache(1000 * kint16max, false));

    unique_ptr<FrontendInterface> base_feature;

    base_feature.reset(
        new ppspeech::Fbank(opts.fbank_opts, std::move(data_source)));

    // CHECK_NE(opts.cmvn_file, "");
    unique_ptr<FrontendInterface> cache;
    if (opts.cmvn_file != ""){
        unique_ptr<FrontendInterface> cmvn(
            new ppspeech::CMVN(opts.cmvn_file, std::move(base_feature)));

        cache.reset(
            new ppspeech::FeatureCache(kint16max, std::move(cmvn)));
    } else {
        cache.reset(
            new ppspeech::FeatureCache(kint16max, std::move(base_feature)));
    }

    base_extractor_.reset(
        new ppspeech::Assembler(opts.assembler_opts, std::move(cache)));
}

}  // namespace ppspeech
