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

#include "frontend/assembler.h"

namespace ppspeech {

using kaldi::BaseFloat;
using std::vector;
using std::vector;
using std::unique_ptr;

Assembler::Assembler(AssemblerOptions opts,
                     unique_ptr<FrontendInterface> base_extractor) {
    fill_zero_ = opts.fill_zero;
    frame_chunk_stride_ = opts.subsampling_rate * opts.nnet_decoder_chunk;
    frame_chunk_size_ = (opts.nnet_decoder_chunk - 1) * opts.subsampling_rate +
                        opts.receptive_filed_length;
    cache_size_ = frame_chunk_size_ - frame_chunk_stride_;
    receptive_filed_length_ = opts.receptive_filed_length;
    base_extractor_ = std::move(base_extractor);
    dim_ = base_extractor_->Dim();
}

void Assembler::Accept(const std::vector<BaseFloat>& inputs) {
    // read inputs
    base_extractor_->Accept(inputs);
}

// pop feature chunk
bool Assembler::Read(std::vector<BaseFloat>* feats) {
    kaldi::Timer timer;
    bool result = Compute(feats);
    VLOG(1) << "Assembler::Read cost: " << timer.Elapsed() << " sec.";
    return result;
}

// read frame by frame from base_feature_extractor_ into cache_
bool Assembler::Compute(vector<BaseFloat>* feats) {
    // compute and feed frame by frame
    while (feature_cache_.size() < frame_chunk_size_) {
        vector<BaseFloat> feature;
        bool result = base_extractor_->Read(&feature);
        if (result == false || feature.size() == 0) {
            VLOG(3) << "result: " << result
                    << " feature dim: " << feature.size();
            if (IsFinished() == false) {
                VLOG(3) << "finished reading feature. cache size: "
                        << feature_cache_.size();
                return false;
            } else {
                VLOG(3) << "break";
                break;
            }
        }

        CHECK(feature.size() == dim_);
        feature_cache_.push(feature);

        nframes_ += 1;
        VLOG(3) << "nframes: " << nframes_;
    }

    if (feature_cache_.size() < receptive_filed_length_) {
        VLOG(3) << "feature_cache less than receptive_filed_length. "
                << feature_cache_.size() << ": " << receptive_filed_length_;
        return false;
    }

    if (fill_zero_) {
        while (feature_cache_.size() < frame_chunk_size_) {
            vector<BaseFloat> feature(dim_, kaldi::kSetZero);
            nframes_ += 1;
            feature_cache_.push(feature);
        }
    }

    int32 this_chunk_size =
        std::min(static_cast<int32>(feature_cache_.size()), frame_chunk_size_);
    feats->resize(dim_ * this_chunk_size);
    VLOG(3) << "read " << this_chunk_size << " feat.";

    int32 counter = 0;
    while (counter < this_chunk_size) {
        vector<BaseFloat>& val = feature_cache_.front();
        CHECK(val.size() == dim_) << val.size();

        int32 start = counter * dim_;
        std::memcpy(
            feats->data() + start, val.data(), val.size() * sizeof(BaseFloat));

        if (this_chunk_size - counter <= cache_size_) {
            feature_cache_.push(val);
        }

        // val is reference, so we should pop here
        feature_cache_.pop();

        counter++;
    }
    CHECK(feature_cache_.size() == cache_size_);

    return true;
}


void Assembler::Reset() {
    std::queue<std::vector<BaseFloat>> empty;
    std::swap(feature_cache_, empty);
    nframes_ = 0;
    base_extractor_->Reset();
}

}  // namespace ppspeech
