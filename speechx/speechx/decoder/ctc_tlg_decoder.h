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

#include "base/common.h"
#include "decoder/decoder_itf.h"
#include "kaldi/decoder/lattice-faster-online-decoder.h"
#include "util/parse-options.h"


DECLARE_string(graph_path);
DECLARE_string(word_symbol_table);
DECLARE_int32(max_active);
DECLARE_double(beam);
DECLARE_double(lattice_beam);

namespace ppspeech {

struct TLGDecoderOptions {
    kaldi::LatticeFasterDecoderConfig opts{};
    // todo remove later, add into decode resource
    std::string word_symbol_table;
    std::string fst_path;

    static TLGDecoderOptions InitFromFlags() {
        TLGDecoderOptions decoder_opts;
        decoder_opts.word_symbol_table = FLAGS_word_symbol_table;
        decoder_opts.fst_path = FLAGS_graph_path;
        LOG(INFO) << "fst path: " << decoder_opts.fst_path;
        LOG(INFO) << "fst symbole table: " << decoder_opts.word_symbol_table;

        decoder_opts.opts.max_active = FLAGS_max_active;
        decoder_opts.opts.beam = FLAGS_beam;
        decoder_opts.opts.lattice_beam = FLAGS_lattice_beam;
        LOG(INFO) << "LatticeFasterDecoder max active: "
                  << decoder_opts.opts.max_active;
        LOG(INFO) << "LatticeFasterDecoder beam: " << decoder_opts.opts.beam;
        LOG(INFO) << "LatticeFasterDecoder lattice_beam: "
                  << decoder_opts.opts.lattice_beam;

        return decoder_opts;
    }
};

class TLGDecoder : public DecoderBase {
  public:
    explicit TLGDecoder(TLGDecoderOptions opts);
    ~TLGDecoder() = default;

    void InitDecoder();
    void Reset();

    void AdvanceDecode(
        const std::shared_ptr<kaldi::DecodableInterface>& decodable);

    void Decode();

    std::string GetFinalBestPath() override;
    std::string GetPartialResult() override;

    int DecodeLikelihoods(const std::vector<std::vector<BaseFloat>>& probs,
                          std::vector<std::string>& nbest_words);

  protected:
    std::string GetBestPath() override {
        CHECK(false);
        return {};
    }
    std::vector<std::pair<double, std::string>> GetNBestPath() override {
        CHECK(false);
        return {};
    }
    std::vector<std::pair<double, std::string>> GetNBestPath(int n) override {
        CHECK(false);
        return {};
    }

  private:
    void AdvanceDecoding(kaldi::DecodableInterface* decodable);

    std::shared_ptr<kaldi::LatticeFasterOnlineDecoder> decoder_;
    std::shared_ptr<fst::Fst<fst::StdArc>> fst_;
    std::shared_ptr<fst::SymbolTable> word_symbol_table_;
};


}  // namespace ppspeech