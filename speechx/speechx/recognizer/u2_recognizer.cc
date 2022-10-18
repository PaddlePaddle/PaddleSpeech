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

#include "recognizer/u2_recognizer.h"

#include "nnet/u2_nnet.h"

namespace ppspeech {

using kaldi::BaseFloat;
using kaldi::SubVector;
using kaldi::Vector;
using kaldi::VectorBase;
using std::unique_ptr;
using std::vector;

U2Recognizer::U2Recognizer(const U2RecognizerResource& resource)
    : opts_(resource) {
    const FeaturePipelineOptions& feature_opts = resource.feature_pipeline_opts;
    feature_pipeline_.reset(new FeaturePipeline(feature_opts));

    std::shared_ptr<NnetBase> nnet(new U2Nnet(resource.model_opts));

    BaseFloat am_scale = resource.acoustic_scale;
    decodable_.reset(new Decodable(nnet, feature_pipeline_, am_scale));

    CHECK(resource.vocab_path != "");
    decoder_.reset(new CTCPrefixBeamSearch(
        resource.vocab_path, resource.decoder_opts.ctc_prefix_search_opts));

    unit_table_ = decoder_->VocabTable();
    symbol_table_ = unit_table_;

    input_finished_ = false;

    Reset();
}

void U2Recognizer::Reset() {
    global_frame_offset_ = 0;
    num_frames_ = 0;
    result_.clear();

    feature_pipeline_->Reset();
    decodable_->Reset();
    decoder_->Reset();
}

void U2Recognizer::ResetContinuousDecoding() {
    global_frame_offset_ = num_frames_;
    num_frames_ = 0;
    result_.clear();

    feature_pipeline_->Reset();
    decodable_->Reset();
    decoder_->Reset();
}


void U2Recognizer::Accept(const VectorBase<BaseFloat>& waves) {
    feature_pipeline_->Accept(waves);
}


void U2Recognizer::Decode() {
    decoder_->AdvanceDecode(decodable_);
    UpdateResult(false);
}

void U2Recognizer::Rescoring() {
    // Do attention Rescoring
    kaldi::Timer timer;
    AttentionRescoring();
    VLOG(1) << "Rescoring cost latency: " << timer.Elapsed() << " sec.";
}

void U2Recognizer::UpdateResult(bool finish) {
    const auto& hypotheses = decoder_->Outputs();
    const auto& inputs = decoder_->Inputs();
    const auto& likelihood = decoder_->Likelihood();
    const auto& times = decoder_->Times();
    result_.clear();

    CHECK_EQ(hypotheses.size(), likelihood.size());
    for (size_t i = 0; i < hypotheses.size(); i++) {
        const std::vector<int>& hypothesis = hypotheses[i];

        DecodeResult path;
        path.score = likelihood[i];
        for (size_t j = 0; j < hypothesis.size(); j++) {
            std::string word = symbol_table_->Find(hypothesis[j]);
            // A detailed explanation of this if-else branch can be found in
            // https://github.com/wenet-e2e/wenet/issues/583#issuecomment-907994058
            if (decoder_->Type() == kWfstBeamSearch) {
                path.sentence += (" " + word);
            } else {
                path.sentence += (word);
            }
        }

        // TimeStamp is only supported in final result
        // TimeStamp of the output of CtcWfstBeamSearch may be inaccurate due to
        // various FST operations when building the decoding graph. So here we
        // use time stamp of the input(e2e model unit), which is more accurate,
        // and it requires the symbol table of the e2e model used in training.
        if (unit_table_ != nullptr && finish) {
            int offset = global_frame_offset_ * FrameShiftInMs();

            const std::vector<int>& input = inputs[i];
            const std::vector<int> time_stamp = times[i];
            CHECK_EQ(input.size(), time_stamp.size());

            for (size_t j = 0; j < input.size(); j++) {
                std::string word = unit_table_->Find(input[j]);

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

void U2Recognizer::AttentionRescoring() {
    decoder_->FinalizeSearch();
    UpdateResult(true);

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

    kaldi::Timer timer;
    std::vector<float> rescoring_score;
    decodable_->AttentionRescoring(
        hypotheses, opts_.decoder_opts.reverse_weight, &rescoring_score);
    VLOG(1) << "Attention Rescoring takes " << timer.Elapsed() << " sec.";

    // combine ctc score and rescoring score
    for (size_t i = 0; i < num_hyps; i++) {
        VLOG(1) << "hyp " << i << " rescoring_score: " << rescoring_score[i]
                << " ctc_score: " << result_[i].score;
        result_[i].score =
            opts_.decoder_opts.rescoring_weight * rescoring_score[i] +
            opts_.decoder_opts.ctc_weight * result_[i].score;
    }

    std::sort(result_.begin(), result_.end(), DecodeResult::CompareFunc);
    VLOG(1) << "result: " << result_[0].sentence
            << " score: " << result_[0].score;
}

std::string U2Recognizer::GetFinalResult() { return result_[0].sentence; }

std::string U2Recognizer::GetPartialResult() { return result_[0].sentence; }

void U2Recognizer::SetFinished() {
    feature_pipeline_->SetFinished();
    input_finished_ = true;
}


}  // namespace ppspeech