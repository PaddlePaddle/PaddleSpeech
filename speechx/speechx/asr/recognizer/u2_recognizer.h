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
#include "frontend/feature_pipeline.h"
#include "fst/fstlib.h"
#include "fst/symbol-table.h"
#include "nnet/decodable.h"

DECLARE_int32(nnet_decoder_chunk);
DECLARE_int32(num_left_chunks);
DECLARE_double(ctc_weight);
DECLARE_double(rescoring_weight);
DECLARE_double(reverse_weight);
DECLARE_int32(nbest);
DECLARE_int32(blank);

DECLARE_double(acoustic_scale);
DECLARE_string(vocab_path);

namespace ppspeech {

struct DecodeOptions {
    // chunk_size is the frame number of one chunk after subsampling.
    // e.g. if subsample rate is 4 and chunk_size = 16, the frames in
    // one chunk are 67=16*4 + 3, stride is 64=16*4
    int chunk_size{16};
    int num_left_chunks{-1};

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
    float ctc_weight{0.0};
    float rescoring_weight{1.0};
    float reverse_weight{0.0};

    // CtcEndpointConfig ctc_endpoint_opts;
    CTCBeamSearchOptions ctc_prefix_search_opts{};

    static DecodeOptions InitFromFlags() {
        DecodeOptions decoder_opts;
        decoder_opts.chunk_size = FLAGS_nnet_decoder_chunk;
        decoder_opts.num_left_chunks = FLAGS_num_left_chunks;
        decoder_opts.ctc_weight = FLAGS_ctc_weight;
        decoder_opts.rescoring_weight = FLAGS_rescoring_weight;
        decoder_opts.reverse_weight = FLAGS_reverse_weight;
        decoder_opts.ctc_prefix_search_opts.blank = FLAGS_blank;
        decoder_opts.ctc_prefix_search_opts.first_beam_size = FLAGS_nbest;
        decoder_opts.ctc_prefix_search_opts.second_beam_size = FLAGS_nbest;
        LOG(INFO) << "chunk_size: " << decoder_opts.chunk_size;
        LOG(INFO) << "num_left_chunks: " << decoder_opts.num_left_chunks;
        LOG(INFO) << "ctc_weight: " << decoder_opts.ctc_weight;
        LOG(INFO) << "rescoring_weight: " << decoder_opts.rescoring_weight;
        LOG(INFO) << "reverse_weight: " << decoder_opts.reverse_weight;
        LOG(INFO) << "blank: " << FLAGS_blank;
        LOG(INFO) << "first_beam_size: " << FLAGS_nbest;
        LOG(INFO) << "second_beam_size: " << FLAGS_nbest;
        return decoder_opts;
    }
};

struct U2RecognizerResource {
    kaldi::BaseFloat acoustic_scale{1.0};
    std::string vocab_path{};

    FeaturePipelineOptions feature_pipeline_opts{};
    ModelOptions model_opts{};
    DecodeOptions decoder_opts{};

    static U2RecognizerResource InitFromFlags() {
        U2RecognizerResource resource;
        resource.vocab_path = FLAGS_vocab_path;
        resource.acoustic_scale = FLAGS_acoustic_scale;
        LOG(INFO) << "vocab path: " << resource.vocab_path;
        LOG(INFO) << "acoustic_scale: " << resource.acoustic_scale;

        resource.feature_pipeline_opts =
            ppspeech::FeaturePipelineOptions::InitFromFlags();
        resource.feature_pipeline_opts.assembler_opts.fill_zero = false;
        LOG(INFO) << "u2 need fill zero be false: "
                  << resource.feature_pipeline_opts.assembler_opts.fill_zero;
        resource.model_opts = ppspeech::ModelOptions::InitFromFlags();
        resource.decoder_opts = ppspeech::DecodeOptions::InitFromFlags();
        return resource;
    }
};


class U2Recognizer {
  public:
    explicit U2Recognizer(const U2RecognizerResource& resouce);
    explicit U2Recognizer(const U2RecognizerResource& resource,
                         std::shared_ptr<NnetBase> nnet);
    ~U2Recognizer();
    void InitDecoder();
    void ResetContinuousDecoding();

    void Accept(const std::vector<kaldi::BaseFloat>& waves);
    void Decode();
    void Rescoring();

    std::string GetFinalResult();
    std::string GetPartialResult();

    void SetInputFinished();
    bool IsFinished() { return input_finished_; }
    void WaitDecodeFinished();
    void WaitFinished();

    bool DecodedSomething() const {
        return !result_.empty() && !result_[0].sentence.empty();
    }

    int FrameShiftInMs() const {
        // one decoder frame length in ms, todo
        return 1;
        //    return decodable_->Nnet()->SubsamplingRate() *
        //          feature_pipeline_->FrameShift();
    }

    const std::vector<DecodeResult>& Result() const { return result_; }
    void AttentionRescoring();

  private:
    static void RunDecoderSearch(U2Recognizer* me);
    void RunDecoderSearchInternal();
    void UpdateResult(bool finish = false);

  private:
    U2RecognizerResource opts_;

    std::shared_ptr<NnetProducer> nnet_producer_;
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
    std::thread thread_;
};

}  // namespace ppspeech