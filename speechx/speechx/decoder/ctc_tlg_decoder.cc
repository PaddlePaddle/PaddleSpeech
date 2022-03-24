#include "decoder/ctc_tlg_decoder.h"
namespace ppspeech {

TLGDecoder::TLGDecoder(TLGDecoderOptions opts) {
    fst_.reset(fst::Fst<fst::StdArc>::Read(opts.fst_path));
    CHECK(fst_ != nullptr);
    word_symbol_table_.reset(fst::SymbolTable::ReadText(opts.word_symbol_table));
    decoder_.reset(new kaldi::LatticeFasterOnlineDecoder(*fst_, opts.opts));
    decoder_->InitDecoding();
}

void TLGDecoder::InitDecoder() {
    decoder_->InitDecoding();
}

void TLGDecoder::AdvanceDecode(const std::shared_ptr<kaldi::DecodableInterface>& decodable) {
    while (1) {
      AdvanceDecoding(decodable.get());
      if (decodable->IsLastFrame(num_frame_decoded_)) break;
    }
}

void TLGDecoder::AdvanceDecoding(kaldi::DecodableInterface* decodable) {
  // skip blank frame?
  decoder_->AdvanceDecoding(decodable, 1);
  num_frame_decoded_++;
}

void TLGDecoder::Reset() {
  decoder_->InitDecoding();
  return;
}

std::string TLGDecoder::GetFinalBestPath() {
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