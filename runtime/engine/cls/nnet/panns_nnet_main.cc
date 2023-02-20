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

#include <fstream>
#include <string>
#include "base/flags.h"
#include "cls/nnet/panns_interface.h"

DEFINE_string(conf_path, "", "config path");
DEFINE_string(scp_path, "", "wav scp path");
DEFINE_string(topk, "", "print topk results");

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = 1;
    CHECK_GT(FLAGS_conf_path.size(), 0);
    CHECK_GT(FLAGS_scp_path.size(), 0);
    CHECK_GT(FLAGS_topk.size(), 0);
    void* instance = ppspeech::ClsCreateInstance(FLAGS_conf_path.c_str());
    int ret = 0;
    // read wav
    std::ifstream ifs(FLAGS_scp_path);
    std::string line = "";
    int topk = std::atoi(FLAGS_topk.c_str());
    while (getline(ifs, line)) {
        // read wav
        char result[1024] = {0};
        ret = ppspeech::ClsFeedForward(
            instance, line.c_str(), topk, result, 1024);
        printf("%s %s\n", line.c_str(), result);
        ret = ppspeech::ClsReset(instance);
    }
    ret = ppspeech::ClsDestroyInstance(instance);
    return 0;
}
