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
#include "utils/picojson.h"

DEFINE_string(json_file, "", "cmvn json file");
DEFINE_string(cmvn_write_path, "./cmvn.ark", "write cmvn");
DEFINE_bool(binary, true, "write cmvn in binary (true) or text(false)");

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;

    LOG(INFO) << "cmvn josn path: " << FLAGS_json_file;

    auto ifs = std::ifstream(FLAGS_json_file);
    std::string json_str = ppspeech::ReadFile2String(FLAGS_json_file);
    picojson::value value;
    std::string err;
    const char* json_end = picojson::parse(
        value, json_str.c_str(), json_str.c_str() + json_str.size(), &err);
    if (!value.is<picojson::object>()) {
        LOG(ERROR) << "Input json file format error.";
    }

    const picojson::value::object& obj = value.get<picojson::object>();
    for (picojson::value::object::const_iterator elem = obj.begin();
         elem != obj.end();
         ++elem) {
        if (elem->first == "mean_stat") {
            VLOG(2) << "mean_stat:" << elem->second;
            // const picojson::value tmp =
            // elem->second.get(0);//<picojson::array>();
            double tmp =
                elem->second.get(0).get<double>();  //<picojson::array>();
            VLOG(2) << "tmp: " << tmp;
        }
        if (elem->first == "var_stat") {
            VLOG(2) << "var_stat: " << elem->second;
        }
        if (elem->first == "frame_num") {
            VLOG(2) << "frame_num: " << elem->second;
        }
    }

    const picojson::value::array& mean_stat =
        value.get("mean_stat").get<picojson::array>();
    std::vector<kaldi::BaseFloat> mean_stat_vec;
    for (auto it = mean_stat.begin(); it != mean_stat.end(); it++) {
        mean_stat_vec.push_back((*it).get<double>());
    }

    const picojson::value::array& var_stat =
        value.get("var_stat").get<picojson::array>();
    std::vector<kaldi::BaseFloat> var_stat_vec;
    for (auto it = var_stat.begin(); it != var_stat.end(); it++) {
        var_stat_vec.push_back((*it).get<double>());
    }

    kaldi::int32 frame_num = value.get("frame_num").get<int64_t>();
    LOG(INFO) << "nframe: " << frame_num;

    size_t mean_size = mean_stat_vec.size();
    kaldi::Matrix<double> cmvn_stats(2, mean_size + 1);
    for (size_t idx = 0; idx < mean_size; ++idx) {
        cmvn_stats(0, idx) = mean_stat_vec[idx];
        cmvn_stats(1, idx) = var_stat_vec[idx];
    }
    cmvn_stats(0, mean_size) = frame_num;
    VLOG(2) << cmvn_stats;

    kaldi::WriteKaldiObject(cmvn_stats, FLAGS_cmvn_write_path, FLAGS_binary);
    LOG(INFO) << "cmvn stats have write into: " << FLAGS_cmvn_write_path;
    LOG(INFO) << "Binary: " << FLAGS_binary;
    return 0;
}
