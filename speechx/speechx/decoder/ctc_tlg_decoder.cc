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

#include "decoder/ctc_tlg_decoder.h"
namespace ppspeech {

TLGDecoder::TLGDecoder(TLGDecoderOptions opts) {
    fst_.reset(fst::Fst<fst::StdArc>::Read(opts.fst_path));
    CHECK(fst_ != nullptr);
    word_symbol_table_.reset(
        fst::SymbolTable::ReadText(opts.word_symbol_table));
    decoder_.reset(new kaldi::LatticeFasterOnlineDecoder(*fst_, opts.opts));
    decoder_->InitDecoding();
    frame_decoded_size_ = 0;
}

void TLGDecoder::InitDecoder() {
    decoder_->InitDecoding();
    frame_decoded_size_ = 0;
}

void TLGDecoder::AdvanceDecode(
    const std::shared_ptr<kaldi::DecodableInterface>& decodable) {
    while (!decodable->IsLastFrame(frame_decoded_size_)) {
        AdvanceDecoding(decodable.get());
    }
}

void TLGDecoder::AdvanceDecoding(kaldi::DecodableInterface* decodable) {
    decoder_->AdvanceDecoding(decodable, 1);
    frame_decoded_size_++;
}

void TLGDecoder::Reset() {
    InitDecoder();
    return;
}

std::string TLGDecoder::GetPartialResult() {
    if (frame_decoded_size_ == 0) {
        // Assertion failed: (this->NumFramesDecoded() > 0 && "You cannot call
        // BestPathEnd if no frames were decoded.")
        return std::string("");
    }
    kaldi::Lattice lat;
    kaldi::LatticeWeight weight;
    std::vector<int> alignment;
    std::vector<int> words_id;
    decoder_->GetBestPath(&lat, false);
    fst::GetLinearSymbolSequence(lat, &alignment, &words_id, &weight);
    std::string words;
    for (int32 idx = 0; idx < words_id.size(); ++idx) {
        std::string word = word_symbol_table_->Find(words_id[idx]);
        words += word;
    }
    return words; 
}

std::string TLGDecoder::GetFinalBestPath() {
    if (frame_decoded_size_ == 0) {
        // Assertion failed: (this->NumFramesDecoded() > 0 && "You cannot call
        // BestPathEnd if no frames were decoded.")
        return std::string("");
    }

    decoder_->FinalizeDecoding();
    kaldi::Lattice lat;
    kaldi::LatticeWeight weight;
    std::vector<int> alignment;
    std::vector<int> words_id;
    decoder_->GetBestPath(&lat, true);
    fst::GetLinearSymbolSequence(lat, &alignment, &words_id, &weight);
    std::string words;
    for (int32 idx = 0; idx < words_id.size(); ++idx) {
        std::string word = word_symbol_table_->Find(words_id[idx]);
        words += word;
    }
    return words;
}
}
