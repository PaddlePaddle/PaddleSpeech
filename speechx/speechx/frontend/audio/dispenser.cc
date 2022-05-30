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

#include "frontend/audio/dispenser.h"

namespace ppspeech {

using kaldi::Vector;
using kaldi::VectorBase;
using kaldi::BaseFloat;
using std::unique_ptr;

Dispenser::Dispenser(DispenserOptions opts,
                     unique_ptr<FrontendInterface> base_extractor) {
    frame_chunk_stride_ = opts.frame_chunk_stride;
    frame_chunk_size_ = opts.frame_chunk_size;
    base_extractor_ = std::move(base_extractor);
    dim_ = base_extractor_->Dim();
}

void Dispenser::Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs) {
    // read inputs
    base_extractor_->Accept(inputs);
}

// pop feature chunk
bool Dispenser::Read(kaldi::Vector<kaldi::BaseFloat>* feats) {
    feats->Resize(dim_ * frame_chunk_size_);
    bool result = Compute(feats);
    return result;
}

// read all data from base_feature_extractor_ into cache_
bool Dispenser::Compute(Vector<BaseFloat>* feats) {
    // compute and feed
    bool result = false;
    while (feature_cache_.size() < frame_chunk_size_) {
        Vector<BaseFloat> feature;
        result = base_extractor_->Read(&feature);
        if (result == false || feature.Dim() == 0) return false;
        feature_cache_.push(feature);
    }

    int32 counter = 0; 
    int32 cache_size = frame_chunk_size_ - frame_chunk_stride_;
    int32 elem_dim = base_extractor_->Dim();
    while (counter < frame_chunk_size_) {
      Vector<BaseFloat>& val = feature_cache_.front();
      int32 start = counter * elem_dim;
      feats->Range(start, elem_dim).CopyFromVec(val);
      if (frame_chunk_size_ - counter <= cache_size ) {
          feature_cache_.push(val);
      }
      feature_cache_.pop();
      counter++;
    }

    return result;
}

}  // namespace ppspeech