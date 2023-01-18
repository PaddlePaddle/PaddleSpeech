/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <fstream>
#include "nnet/cls_interface.h"

int main(int argc, char* argv[]) {
    if (argc != 4){
        printf("usage : cls_nnet_main conf_path wav_path topk\n");
        return 0;
    }
    const char* conf_path = argv[1];
    const char* wav_path = argv[2];
    int topk = std::atoi(argv[3]);
    void* instance = ppspeech::cls_create_instance(conf_path);
    int ret = 0;
    //read wav
    std::ifstream ifs(wav_path);
    std::string line = "";
    while(getline(ifs, line)){
        //read wav
        char result[1024] = {0};
        ret = ppspeech::cls_feedforward(instance, line.c_str(), topk, result, 1024);
        printf("%s %s\n", line.c_str(), result);
        ret = ppspeech::cls_reset(instance);
    }
    ret = ppspeech::cls_destroy_instance(instance);
    return 0;
}
