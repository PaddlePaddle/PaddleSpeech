#pragma once

#include "kaldi/decoder/lattice-faster-online-decoder.h"
#include "kaldi/decoder/decodable-itf.h"
#include "util/parse-options.h"
#include "base/basic_types.h"

namespace ppspeech {

struct TLGDecoderOptions {
   kaldi::LatticeFasterDecoderConfig opts; 
   // todo remove later, add into decode resource 
   std::string word_symbol_table;
   std::string fst_path;
   
   TLGDecoderOptions() 
       : word_symbol_table(""),
         fst_path("") {}
};

class TLGDecoder {
  public:
    explicit TLGDecoder(TLGDecoderOptions opts);
    void InitDecoder();
    void Decode();
    std::string GetBestPath();
    std::vector<std::pair<double, std::string>> GetNBestPath();
    std::string GetFinalBestPath();
    int NumFrameDecoded();
    int DecodeLikelihoods(const std::vector<std::vector<BaseFloat>>& probs,
                          std::vector<std::string>& nbest_words);
    void AdvanceDecode(
        const std::shared_ptr<kaldi::DecodableInterface>& decodable);
    void Reset();

  private:
    void AdvanceDecoding(kaldi::DecodableInterface* decodable);    

    std::shared_ptr<kaldi::LatticeFasterOnlineDecoder> decoder_;
    std::shared_ptr<fst::Fst<fst::StdArc>> fst_;   
    std::shared_ptr<fst::SymbolTable> word_symbol_table_;
    int32 num_frame_decoded_;
  };

    

}  // namespace ppspeech