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

#include "frontend/audio/assembler.h"

namespace ppspeech {

using kaldi::BaseFloat;
using kaldi::Vector;
using kaldi::VectorBase;
using std::unique_ptr;

Assembler::Assembler(AssemblerOptions opts,
                     unique_ptr<FrontendInterface> base_extractor) {
    frame_chunk_stride_ = opts.subsampling_rate * opts.nnet_decoder_chunk;
    frame_chunk_size_ = (opts.nnet_decoder_chunk - 1) * opts.subsampling_rate +
                        opts.receptive_filed_length;
    receptive_filed_length_ = opts.receptive_filed_length;
    base_extractor_ = std::move(base_extractor);
    dim_ = base_extractor_->Dim();
}

void Assembler::Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs) {
    // read inputs
    base_extractor_->Accept(inputs);
}

// pop feature chunk
bool Assembler::Read(kaldi::Vector<kaldi::BaseFloat>* feats) {
    feats->Resize(dim_ * frame_chunk_size_);
    bool result = Compute(feats);
    return result;
}

// read all data from base_feature_extractor_ into cache_
bool Assembler::Compute(Vector<BaseFloat>* feats) {
    // compute and feed
    bool result = false;
    while (feature_cache_.size() < frame_chunk_size_) {
        Vector<BaseFloat> feature;
        result = base_extractor_->Read(&feature);
        if (result == false || feature.Dim() == 0) {
            if (IsFinished() == false) return false;
            break;
        }
        feature_cache_.push(feature);
    }

    if (feature_cache_.size() < receptive_filed_length_) {
        return false;
    }

    while (feature_cache_.size() < frame_chunk_size_) {
        Vector<BaseFloat> feature(dim_, kaldi::kSetZero);
        feature_cache_.push(feature);
    }

    int32 counter = 0;
    int32 cache_size = frame_chunk_size_ - frame_chunk_stride_;
    int32 elem_dim = base_extractor_->Dim();
    while (counter < frame_chunk_size_) {
        Vector<BaseFloat>& val = feature_cache_.front();
        int32 start = counter * elem_dim;
        feats->Range(start, elem_dim).CopyFromVec(val);
        if (frame_chunk_size_ - counter <= cache_size) {
            feature_cache_.push(val);
        }
        feature_cache_.pop();
        counter++;
    }

    return result;
}

}  // namespace ppspeech
