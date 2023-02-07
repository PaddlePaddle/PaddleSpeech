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

#include "decoder/ctc_tlg_decoder.h"
#include "base/common.h"
#include "decoder/param.h"
#include "frontend/data_cache.h"
#include "kaldi/util/table-types.h"
#include "nnet/decodable.h"
#include "nnet/nnet_producer.h"


DEFINE_string(nnet_prob_rspecifier, "", "test feature rspecifier");
DEFINE_string(result_wspecifier, "", "test result wspecifier");


using kaldi::BaseFloat;
using kaldi::Matrix;
using std::vector;

// test TLG decoder by feeding speech feature.
int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    kaldi::SequentialBaseFloatMatrixReader nnet_prob_reader(
        FLAGS_nnet_prob_rspecifier);
    kaldi::TokenWriter result_writer(FLAGS_result_wspecifier);

    int32 num_done = 0, num_err = 0;

    ppspeech::TLGDecoderOptions opts =
        ppspeech::TLGDecoderOptions::InitFromFlags();
    opts.opts.beam = 15.0;
    opts.opts.lattice_beam = 7.5;
    ppspeech::TLGDecoder decoder(opts);

    ppspeech::ModelOptions model_opts = ppspeech::ModelOptions::InitFromFlags();

    std::shared_ptr<ppspeech::NnetProducer> nnet_producer =
        std::make_shared<ppspeech::NnetProducer>(nullptr);
    std::shared_ptr<ppspeech::Decodable> decodable(
        new ppspeech::Decodable(nnet_producer, FLAGS_acoustic_scale));

    decoder.InitDecoder();
    kaldi::Timer timer;

    for (; !nnet_prob_reader.Done(); nnet_prob_reader.Next()) {
        string utt = nnet_prob_reader.Key();
        kaldi::Matrix<BaseFloat> prob = nnet_prob_reader.Value();
        decodable->Acceptlikelihood(prob);
        decoder.AdvanceDecode(decodable);
        std::string result;
        result = decoder.GetFinalBestPath();
        decodable->Reset();
        decoder.Reset();
        if (result.empty()) {
            // the TokenWriter can not write empty string.
            ++num_err;
            KALDI_LOG << " the result of " << utt << " is empty";
            continue;
        }
        KALDI_LOG << " the result of " << utt << " is " << result;
        result_writer.Write(utt, result);
        ++num_done;
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << " cost:" << elapsed << " s";

    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
