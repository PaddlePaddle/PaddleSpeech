

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

#pragma once

#include "decoder/common.h"
#include "decoder/ctc_beam_search_opt.h"
#include "decoder/ctc_prefix_beam_search_decoder.h"
#include "decoder/decoder_itf.h"
#include "frontend/audio/feature_pipeline.h"
#include "nnet/decodable.h"

#include "fst/fstlib.h"
#include "fst/symbol-table.h"

namespace ppspeech {


struct DecodeOptions {
    // chunk_size is the frame number of one chunk after subsampling.
    // e.g. if subsample rate is 4 and chunk_size = 16, the frames in
    // one chunk are 67=16*4 + 3, stride is 64=16*4
    int chunk_size;
    int num_left_chunks;

    // final_score = rescoring_weight * rescoring_score + ctc_weight *
    // ctc_score;
    // rescoring_score = left_to_right_score * (1 - reverse_weight) +
    // right_to_left_score * reverse_weight
    // Please note the concept of ctc_scores
    // in the following two search methods are different. For
    // CtcPrefixBeamSerch,
    // it's a sum(prefix) score + context score For CtcWfstBeamSerch, it's a
    // max(viterbi) path score + context score So we should carefully set
    // ctc_weight accroding to the search methods.
    float ctc_weight;
    float rescoring_weight;
    float reverse_weight;

    // CtcEndpointConfig ctc_endpoint_opts;
    CTCBeamSearchOptions ctc_prefix_search_opts;

    DecodeOptions()
        : chunk_size(16),
          num_left_chunks(-1),
          ctc_weight(0.5),
          rescoring_weight(1.0),
          reverse_weight(0.0) {}

    void Register(kaldi::OptionsItf* opts) {
        std::string module = "DecoderConfig: ";
        opts->Register(
            "chunk-size",
            &chunk_size,
            module + "the frame number of one chunk after subsampling.");
        opts->Register("num-left-chunks",
                       &num_left_chunks,
                       module + "the left history chunks number.");
        opts->Register("ctc-weight",
                       &ctc_weight,
                       module +
                           "ctc weight for rescore. final_score = "
                           "rescoring_weight * rescoring_score + ctc_weight * "
                           "ctc_score.");
        opts->Register("rescoring-weight",
                       &rescoring_weight,
                       module +
                           "attention score weight for rescore. final_score = "
                           "rescoring_weight * rescoring_score + ctc_weight * "
                           "ctc_score.");
        opts->Register("reverse-weight",
                       &reverse_weight,
                       module +
                           "reverse decoder weight. rescoring_score = "
                           "left_to_right_score * (1 - reverse_weight) + "
                           "right_to_left_score * reverse_weight.");
    }
};


struct U2RecognizerResource {
    kaldi::BaseFloat acoustic_scale{1.0};
    std::string vocab_path{};

    FeaturePipelineOptions feature_pipeline_opts{};
    ModelOptions model_opts{};
    DecodeOptions decoder_opts{};
    //    CTCBeamSearchOptions beam_search_opts;
};


class U2Recognizer {
  public:
    explicit U2Recognizer(const U2RecognizerResource& resouce);
    void Reset();
    void ResetContinuousDecoding();

    void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& waves);
    void Decode();
    void Rescoring();


    std::string GetFinalResult();
    std::string GetPartialResult();

    void SetFinished();
    bool IsFinished() { return input_finished_; }

    bool DecodedSomething() const {
        return !result_.empty() && !result_[0].sentence.empty();
    }


    int FrameShiftInMs() const {
        // one decoder frame length in ms
        return decodable_->Nnet()->SubsamplingRate() *
               feature_pipeline_->FrameShift();
    }


    const std::vector<DecodeResult>& Result() const { return result_; }

  private:
    void AttentionRescoring();
    void UpdateResult(bool finish = false);

  private:
    U2RecognizerResource opts_;

    // std::shared_ptr<U2RecognizerResource> resource_;
    // U2RecognizerResource resource_;
    std::shared_ptr<FeaturePipeline> feature_pipeline_;
    std::shared_ptr<Decodable> decodable_;
    std::unique_ptr<CTCPrefixBeamSearch> decoder_;

    // e2e unit symbol table
    std::shared_ptr<fst::SymbolTable> unit_table_ = nullptr;
    std::shared_ptr<fst::SymbolTable> symbol_table_ = nullptr;

    std::vector<DecodeResult> result_;

    // global decoded frame offset
    int global_frame_offset_;
    // cur decoded frame num
    int num_frames_;
    // timestamp gap between words in a sentence
    const int time_stamp_gap_ = 100;

    bool input_finished_;
};

}  // namespace ppspeech