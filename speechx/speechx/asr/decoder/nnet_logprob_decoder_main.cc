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

// todo refactor, repalce with gtest

#include "base/flags.h"
#include "base/log.h"
#include "decoder/ctc_beam_search_decoder.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"

DEFINE_string(nnet_prob_respecifier, "", "test nnet prob rspecifier");
DEFINE_string(dict_file, "vocab.txt", "vocabulary of lm");
DEFINE_string(lm_path, "lm.klm", "language model");

using kaldi::BaseFloat;
using kaldi::Matrix;
using std::vector;

// test decoder by feeding nnet posterior probability
int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    kaldi::SequentialBaseFloatMatrixReader likelihood_reader(
        FLAGS_nnet_prob_respecifier);
    std::string dict_file = FLAGS_dict_file;
    std::string lm_path = FLAGS_lm_path;
    LOG(INFO) << "dict path: " << dict_file;
    LOG(INFO) << "lm path: " << lm_path;

    int32 num_done = 0, num_err = 0;

    ppspeech::CTCBeamSearchOptions opts;
    opts.dict_file = dict_file;
    opts.lm_path = lm_path;
    ppspeech::CTCBeamSearch decoder(opts);

    std::shared_ptr<ppspeech::Decodable> decodable(
        new ppspeech::Decodable(nullptr, nullptr));

    decoder.InitDecoder();

    for (; !likelihood_reader.Done(); likelihood_reader.Next()) {
        string utt = likelihood_reader.Key();
        const kaldi::Matrix<BaseFloat> likelihood = likelihood_reader.Value();
        LOG(INFO) << "process utt: " << utt;
        LOG(INFO) << "rows: " << likelihood.NumRows();
        LOG(INFO) << "cols: " << likelihood.NumCols();
        decodable->Acceptlikelihood(likelihood);
        decoder.AdvanceDecode(decodable);
        std::string result;
        result = decoder.GetFinalBestPath();
        KALDI_LOG << " the result of " << utt << " is " << result;
        decodable->Reset();
        decoder.Reset();
        ++num_done;
    }

    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
