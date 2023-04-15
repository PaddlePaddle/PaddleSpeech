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

#include "recognizer/recognizer_instance.h"


namespace ppspeech {

RecognizerInstance& RecognizerInstance::GetInstance() {
    static RecognizerInstance instance;
    return instance;
}

bool RecognizerInstance::Init(const std::string& model_file, 
                              const std::string& word_symbol_table_file,
                              const std::string& fst_file,
                              int num_instance) {
    RecognizerResource resource = RecognizerResource::InitFromFlags();
    resource.model_opts.model_path = model_file;
    //resource.vocab_path = word_symbol_table_file;
    if (!fst_file.empty()) {
        resource.decoder_opts.tlg_decoder_opts.fst_path = fst_file;
        resource.decoder_opts.tlg_decoder_opts.fst_path = word_symbol_table_file;
    } else {
        resource.decoder_opts.ctc_prefix_search_opts.word_symbol_table = 
            word_symbol_table_file;
    }
    recognizer_controller_ = std::make_unique<RecognizerController>(num_instance, resource);
    return true;
}

void RecognizerInstance::InitDecoder(int idx) {
    recognizer_controller_->InitDecoder(idx);
    return;
}

int RecognizerInstance::GetRecognizerInstanceId() {
    return recognizer_controller_->GetRecognizerInstanceId();
}

void RecognizerInstance::Accept(const std::vector<float>& waves, int idx) const {
    recognizer_controller_->Accept(waves, idx);
    return;
} 

void RecognizerInstance::SetInputFinished(int idx) const {
    recognizer_controller_->SetInputFinished(idx);
    return;
}

std::string RecognizerInstance::GetResult(int idx) const {
    return recognizer_controller_->GetFinalResult(idx);
}

}