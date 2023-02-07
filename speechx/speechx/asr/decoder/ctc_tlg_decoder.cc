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

TLGDecoder::TLGDecoder(TLGDecoderOptions opts) : opts_(opts) {
    fst_.reset(fst::Fst<fst::StdArc>::Read(opts.fst_path));
    CHECK(fst_ != nullptr);

    word_symbol_table_.reset(
        fst::SymbolTable::ReadText(opts.word_symbol_table));

    decoder_.reset(new kaldi::LatticeFasterOnlineDecoder(*fst_, opts.opts));

    Reset();
}

void TLGDecoder::Reset() {
    decoder_->InitDecoding();
    num_frame_decoded_ = 0;
    return;
}

void TLGDecoder::InitDecoder() { Reset(); }

void TLGDecoder::AdvanceDecode(
    const std::shared_ptr<kaldi::DecodableInterface>& decodable) {
    while (!decodable->IsLastFrame(num_frame_decoded_)) {
        AdvanceDecoding(decodable.get());
    }
}

void TLGDecoder::AdvanceDecoding(kaldi::DecodableInterface* decodable) {
    decoder_->AdvanceDecoding(decodable, 1);
    num_frame_decoded_++;
}


std::string TLGDecoder::GetPartialResult() {
    if (num_frame_decoded_ == 0) {
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

void TLGDecoder::FinalizeSearch() {
    decoder_->FinalizeDecoding();
    kaldi::CompactLattice clat;
    decoder_->GetLattice(&clat, true);
    kaldi::Lattice lat, nbest_lat;
    fst::ConvertLattice(clat, &lat);
    fst::ShortestPath(lat, &nbest_lat, opts_.nbest);
    std::vector<kaldi::Lattice> nbest_lats;
    fst::ConvertNbestToVector(nbest_lat, &nbest_lats);

    hypotheses_.clear();
    hypotheses_.reserve(nbest_lats.size());
    likelihood_.clear();
    likelihood_.reserve(nbest_lats.size());
    times_.clear();
    times_.reserve(nbest_lats.size());
    for (auto lat : nbest_lats) {
        kaldi::LatticeWeight weight;
        std::vector<int> hypothese;
        std::vector<int> time;
        std::vector<int> alignment;
        std::vector<int> words_id;
        fst::GetLinearSymbolSequence(lat, &alignment, &words_id, &weight);
        int idx = 0;
        for (; idx < alignment.size() - 1; ++idx) {
            if (alignment[idx] == 0) continue;
            if (alignment[idx] != alignment[idx + 1]) {
                hypothese.push_back(alignment[idx] - 1);
                time.push_back(idx);  // fake time, todo later
            }
        }
        hypothese.push_back(alignment[idx] - 1);
        time.push_back(idx);  // fake time, todo later
        hypotheses_.push_back(hypothese);
        times_.push_back(time);
        olabels.push_back(words_id);
        likelihood_.push_back(-(weight.Value2() + weight.Value1()));
    }
}

std::string TLGDecoder::GetFinalBestPath() {
    if (num_frame_decoded_ == 0) {
        // Assertion failed: (this->NumFramesDecoded() > 0 && "You cannot call
        // BestPathEnd if no frames were decoded.")
        return std::string("");
    }
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

}  // namespace ppspeech
