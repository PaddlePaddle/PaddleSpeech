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

#include "base/flags.h"
#include "base/log.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "kaldi/util/kaldi-io.h"
#include "utils/file_utils.h"
#include "utils/simdjson.h"

DEFINE_string(json_file, "", "cmvn json file");
DEFINE_string(cmvn_write_path, "./cmvn.ark", "write cmvn");
DEFINE_bool(binary, true, "write cmvn in binary (true) or text(false)");

using namespace simdjson;

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    ondemand::parser parser;
    padded_string json = padded_string::load(FLAGS_json_file);
    ondemand::document val = parser.iterate(json);
    ondemand::object doc = val;
    kaldi::int32 frame_num = uint64_t(doc["frame_num"]);
    auto mean_stat = doc["mean_stat"];
    std::vector<kaldi::BaseFloat> mean_stat_vec;
    for (double x : mean_stat) {
        mean_stat_vec.push_back(x);
    }
    auto var_stat = doc["var_stat"];
    std::vector<kaldi::BaseFloat> var_stat_vec;
    for (double x : var_stat) {
        var_stat_vec.push_back(x);
    }

    size_t mean_size = mean_stat_vec.size();
    kaldi::Matrix<double> cmvn_stats(2, mean_size + 1);
    for (size_t idx = 0; idx < mean_size; ++idx) {
        cmvn_stats(0, idx) = mean_stat_vec[idx];
        cmvn_stats(1, idx) = var_stat_vec[idx];
    }
    cmvn_stats(0, mean_size) = frame_num;
    kaldi::WriteKaldiObject(cmvn_stats, FLAGS_cmvn_write_path, FLAGS_binary);
    LOG(INFO) << "the json file have write into " << FLAGS_cmvn_write_path;
    return 0;
}