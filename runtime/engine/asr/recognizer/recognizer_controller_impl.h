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

#pragma once

#include "decoder/common.h"
#include "fst/fstlib.h"
#include "fst/symbol-table.h"
#include "nnet/u2_nnet.h"
#include "nnet/nnet_producer.h"
#ifdef USE_ONNX
#include "nnet/u2_onnx_nnet.h"
#endif
#include "nnet/decodable.h"
#include "recognizer/recognizer_resource.h"

#include <memory>

namespace ppspeech {

class RecognizerControllerImpl {
  public:
    explicit RecognizerControllerImpl(const RecognizerResource& resource);
    explicit RecognizerControllerImpl(const RecognizerResource& resource, 
                                      std::shared_ptr<NnetBase> nnet);
    ~RecognizerControllerImpl();
    void Accept(std::vector<float> data);
    void InitDecoder();
    void SetInputFinished();
    std::string GetFinalResult();
    std::string GetPartialResult();
    void Rescoring();
    void Reset();
    void WaitDecoderFinished();
    void WaitFinished();
    void AttentionRescoring();
    bool DecodedSomething() const {
      return !result_.empty() && !result_[0].sentence.empty();
    }
    int FrameShiftInMs() const {
      return 1; //todo
    }

  private:

    static void RunNnetEvaluation(RecognizerControllerImpl* me);
    void RunNnetEvaluationInternal();
    static void RunDecoder(RecognizerControllerImpl* me);
    void RunDecoderInternal();
    void UpdateResult(bool finish = false);

    std::shared_ptr<Decodable> decodable_;
    std::unique_ptr<DecoderBase> decoder_;
    std::shared_ptr<NnetProducer> nnet_producer_;

    // e2e unit symbol table
    std::shared_ptr<fst::SymbolTable> symbol_table_ = nullptr;
    std::vector<DecodeResult> result_;

    RecognizerResource opts_;
    bool abort_ = false;
    // global decoded frame offset
    int global_frame_offset_;
    // cur decoded frame num
    int num_frames_;
    // timestamp gap between words in a sentence
    const int time_stamp_gap_ = 100;
    bool input_finished_;

    std::mutex nnet_mutex_;
    std::mutex decoder_mutex_;
    std::condition_variable nnet_condition_;
    std::condition_variable decoder_condition_;
    std::thread nnet_thread_;
    std::thread decoder_thread_;

    DISALLOW_COPY_AND_ASSIGN(RecognizerControllerImpl);
};

}