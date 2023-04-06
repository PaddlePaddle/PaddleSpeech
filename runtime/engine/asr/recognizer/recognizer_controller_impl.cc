// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "recognizer/recognizer_controller_impl.h"
#include "decoder/ctc_prefix_beam_search_decoder.h"
#include "common/utils/strings.h"

namespace ppspeech {

RecognizerControllerImpl::RecognizerControllerImpl(const RecognizerResource& resource)
: opts_(resource) {
    BaseFloat am_scale = resource.acoustic_scale;
    const FeaturePipelineOptions& feature_opts = resource.feature_pipeline_opts;
    std::shared_ptr<FeaturePipeline> feature_pipeline(
        new FeaturePipeline(feature_opts));
    std::shared_ptr<NnetBase> nnet;
#ifndef USE_ONNX
    nnet.reset(new U2Nnet(resource.model_opts));
#else
    if (resource.model_opts.with_onnx_model){
        nnet.reset(new U2OnnxNnet(resource.model_opts));
    } else {
        nnet.reset(new U2Nnet(resource.model_opts));
    }
#endif
    nnet_producer_.reset(new NnetProducer(nnet, feature_pipeline));
    nnet_thread_ = std::thread(RunNnetEvaluation, this);

    decodable_.reset(new Decodable(nnet_producer_, am_scale));
    CHECK_NE(resource.vocab_path, "");
    if (resource.decoder_opts.tlg_decoder_opts.fst_path.empty()) {
        LOG(INFO) << resource.decoder_opts.tlg_decoder_opts.fst_path;
        decoder_ = std::make_unique<CTCPrefixBeamSearch>(
            resource.vocab_path, resource.decoder_opts.ctc_prefix_search_opts);
    } else {
        decoder_ = std::make_unique<TLGDecoder>(
            resource.decoder_opts.tlg_decoder_opts);
    }

    symbol_table_ = decoder_->WordSymbolTable();
    global_frame_offset_ = 0;
    input_finished_ = false;
    num_frames_ = 0;
    result_.clear(); 
}

RecognizerControllerImpl::RecognizerControllerImpl(const RecognizerResource& resource,
                                                   std::shared_ptr<NnetBase> nnet)
    :opts_(resource) {
    BaseFloat am_scale = resource.acoustic_scale;
    const FeaturePipelineOptions& feature_opts = resource.feature_pipeline_opts;
    std::shared_ptr<FeaturePipeline> feature_pipeline =
        std::make_shared<FeaturePipeline>(feature_opts);
    nnet_producer_ = std::make_shared<NnetProducer>(nnet, feature_pipeline);
    nnet_thread_ = std::thread(RunNnetEvaluation, this);
    decodable_.reset(new Decodable(nnet_producer_, am_scale));

    CHECK_NE(resource.vocab_path, "");
    if (resource.decoder_opts.tlg_decoder_opts.fst_path == "") {
        decoder_.reset(new CTCPrefixBeamSearch(
            resource.vocab_path, resource.decoder_opts.ctc_prefix_search_opts));
    } else {
        decoder_.reset(new TLGDecoder(resource.decoder_opts.tlg_decoder_opts));
    }

    symbol_table_ = decoder_->WordSymbolTable();

    global_frame_offset_ = 0;
    input_finished_ = false;
    num_frames_ = 0;
    result_.clear();
}

RecognizerControllerImpl::~RecognizerControllerImpl() {
    WaitFinished();
}

void RecognizerControllerImpl::Reset() {
    nnet_producer_->Reset();
}

void RecognizerControllerImpl::RunDecoder(RecognizerControllerImpl* me) {
    me->RunDecoderInternal();
}

void RecognizerControllerImpl::RunDecoderInternal() {
    LOG(INFO) << "DecoderInternal begin";
    while (!nnet_producer_->IsFinished()) {
        nnet_condition_.notify_one();
        decoder_->AdvanceDecode(decodable_);
    }
    decoder_->AdvanceDecode(decodable_);
    UpdateResult(false);
    LOG(INFO) << "DecoderInternal exit";
}

void RecognizerControllerImpl::WaitDecoderFinished() {
    if (decoder_thread_.joinable()) decoder_thread_.join();
}

void RecognizerControllerImpl::RunNnetEvaluation(RecognizerControllerImpl* me) {
    me->RunNnetEvaluationInternal();
}

void RecognizerControllerImpl::SetInputFinished() {
    nnet_producer_->SetInputFinished();
    nnet_condition_.notify_one();
    LOG(INFO) << "Set Input Finished";
}

void RecognizerControllerImpl::WaitFinished() {
    abort_ = true;
    LOG(INFO) << "nnet wait finished";
    nnet_condition_.notify_one();
    if (nnet_thread_.joinable()) {
        nnet_thread_.join();
    }
}

void RecognizerControllerImpl::RunNnetEvaluationInternal() {
    bool result = false;
    LOG(INFO) << "NnetEvaluationInteral begin";
    while (!abort_) {
        std::unique_lock<std::mutex> lock(nnet_mutex_);
        nnet_condition_.wait(lock);
        do {
            result = nnet_producer_->Compute();
            decoder_condition_.notify_one();
        } while (result);
    }
    LOG(INFO) << "NnetEvaluationInteral exit";    
}

void RecognizerControllerImpl::Accept(std::vector<float> data) {
    nnet_producer_->Accept(data);
    nnet_condition_.notify_one();
}

void RecognizerControllerImpl::InitDecoder() {
    global_frame_offset_ = 0;
    input_finished_ = false;
    num_frames_ = 0;
    result_.clear();

    decodable_->Reset();
    decoder_->Reset();
    decoder_thread_ = std::thread(RunDecoder, this);
}

void RecognizerControllerImpl::AttentionRescoring() {
    decoder_->FinalizeSearch();
    UpdateResult(false);

    // No need to do rescoring
    if (0.0 == opts_.decoder_opts.rescoring_weight) {
        LOG_EVERY_N(WARNING, 3) << "Not do AttentionRescoring!";
        return;
    }
    LOG_EVERY_N(WARNING, 3) << "Do AttentionRescoring!";

    // Inputs() returns N-best input ids, which is the basic unit for rescoring
    // In CtcPrefixBeamSearch, inputs are the same to outputs
    const auto& hypotheses = decoder_->Inputs();
    int num_hyps = hypotheses.size();
    if (num_hyps <= 0) {
        return;
    }

    std::vector<float> rescoring_score;
    decodable_->AttentionRescoring(
        hypotheses, opts_.decoder_opts.reverse_weight, &rescoring_score);

    // combine ctc score and rescoring score
    for (size_t i = 0; i < num_hyps; i++) {
        VLOG(3) << "hyp " << i << " rescoring_score: " << rescoring_score[i]
                << " ctc_score: " << result_[i].score
                << " rescoring_weight: " << opts_.decoder_opts.rescoring_weight
                << " ctc_weight: " << opts_.decoder_opts.ctc_weight;
        result_[i].score =
            opts_.decoder_opts.rescoring_weight * rescoring_score[i] +
            opts_.decoder_opts.ctc_weight * result_[i].score;

        VLOG(3) << "hyp: " << result_[0].sentence
                << " score: " << result_[0].score;
    }

    std::sort(result_.begin(), result_.end(), DecodeResult::CompareFunc);
    VLOG(3) << "result: " << result_[0].sentence
            << " score: " << result_[0].score;
}

std::string RecognizerControllerImpl::GetFinalResult() { return result_[0].sentence; }

std::string RecognizerControllerImpl::GetPartialResult() { return result_[0].sentence; }

void RecognizerControllerImpl::UpdateResult(bool finish) {
    const auto& hypotheses = decoder_->Outputs();
    const auto& inputs = decoder_->Inputs();
    const auto& likelihood = decoder_->Likelihood();
    const auto& times = decoder_->Times();
    result_.clear();

    CHECK_EQ(inputs.size(), likelihood.size());
    for (size_t i = 0; i < hypotheses.size(); i++) {
        const std::vector<int>& hypothesis = hypotheses[i];

        DecodeResult path;
        path.score = likelihood[i];
        for (size_t j = 0; j < hypothesis.size(); j++) {
            std::string word = symbol_table_->Find(hypothesis[j]);
            path.sentence += (" " + word);
        }
        path.sentence = DelBlank(path.sentence);

        // TimeStamp is only supported in final result
        // TimeStamp of the output of CtcWfstBeamSearch may be inaccurate due to
        // various FST operations when building the decoding graph. So here we
        // use time stamp of the input(e2e model unit), which is more accurate,
        // and it requires the symbol table of the e2e model used in training.
        if (symbol_table_ != nullptr && finish) {
            int offset = global_frame_offset_ * FrameShiftInMs();

            const std::vector<int>& input = inputs[i];
            const std::vector<int> time_stamp = times[i];
            CHECK_EQ(input.size(), time_stamp.size());

            for (size_t j = 0; j < input.size(); j++) {
                std::string word = symbol_table_->Find(input[j]);

                int start =
                    time_stamp[j] * FrameShiftInMs() - time_stamp_gap_ > 0
                        ? time_stamp[j] * FrameShiftInMs() - time_stamp_gap_
                        : 0;
                if (j > 0) {
                    start =
                        (time_stamp[j] - time_stamp[j - 1]) * FrameShiftInMs() <
                                time_stamp_gap_
                            ? (time_stamp[j - 1] + time_stamp[j]) / 2 *
                                  FrameShiftInMs()
                            : start;
                }

                int end = time_stamp[j] * FrameShiftInMs();
                if (j < input.size() - 1) {
                    end =
                        (time_stamp[j + 1] - time_stamp[j]) * FrameShiftInMs() <
                                time_stamp_gap_
                            ? (time_stamp[j + 1] + time_stamp[j]) / 2 *
                                  FrameShiftInMs()
                            : end;
                }

                WordPiece word_piece(word, offset + start, offset + end);
                path.word_pieces.emplace_back(word_piece);
            }
        }

        // if (post_processor_ != nullptr) {
        //   path.sentence = post_processor_->Process(path.sentence, finish);
        // }

        result_.emplace_back(path);
    }

    if (DecodedSomething()) {
        VLOG(1) << "Partial CTC result " << result_[0].sentence;
    }
}

}  // namespace ppspeech