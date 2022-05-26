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

// Note: Do not print/log ondemand object.

#include "base/common.h"
#include "base/flags.h"
#include "base/log.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "kaldi/util/kaldi-io.h"
#include "utils/file_utils.h"
// #include "boost/json.hpp"
#include <boost/json/src.hpp>

DEFINE_string(json_file, "", "cmvn json file");
DEFINE_string(cmvn_write_path, "./cmvn.ark", "write cmvn");
DEFINE_bool(binary, true, "write cmvn in binary (true) or text(false)");

using namespace boost::json;  // from <boost/json.hpp>

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "cmvn josn path: " << FLAGS_json_file;

    auto ifs = std::ifstream(FLAGS_json_file);
    std::string json_str = ppspeech::ReadFile2String(FLAGS_json_file);
    auto value = boost::json::parse(json_str);
    if (!value.is_object()) {
        LOG(ERROR) << "Input json file format error.";
    }

    for (auto obj : value.as_object()) {
        if (obj.key() == "mean_stat") {
            LOG(INFO) << "mean_stat:" << obj.value();
        }
        if (obj.key() == "var_stat") {
            LOG(INFO) << "var_stat: " << obj.value();
        }
        if (obj.key() == "frame_num") {
            LOG(INFO) << "frame_num: " << obj.value();
        }
    }

    boost::json::array mean_stat = value.at("mean_stat").as_array();
    std::vector<kaldi::BaseFloat> mean_stat_vec;
    for (auto it = mean_stat.begin(); it != mean_stat.end(); it++) {
        mean_stat_vec.push_back(it->as_double());
    }

    boost::json::array var_stat = value.at("var_stat").as_array();
    std::vector<kaldi::BaseFloat> var_stat_vec;
    for (auto it = var_stat.begin(); it != var_stat.end(); it++) {
        var_stat_vec.push_back(it->as_double());
    }

    kaldi::int32 frame_num = uint64_t(value.at("frame_num").as_int64());
    LOG(INFO) << "nframe: " << frame_num;

    size_t mean_size = mean_stat_vec.size();
    kaldi::Matrix<double> cmvn_stats(2, mean_size + 1);
    for (size_t idx = 0; idx < mean_size; ++idx) {
        cmvn_stats(0, idx) = mean_stat_vec[idx];
        cmvn_stats(1, idx) = var_stat_vec[idx];
    }
    cmvn_stats(0, mean_size) = frame_num;
    LOG(INFO) << cmvn_stats;

    kaldi::WriteKaldiObject(cmvn_stats, FLAGS_cmvn_write_path, FLAGS_binary);
    LOG(INFO) << "cmvn stats have write into: " << FLAGS_cmvn_write_path;
    LOG(INFO) << "Binary: " << FLAGS_binary;
    return 0;
}
