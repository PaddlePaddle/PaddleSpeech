// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "recognizer.h"
#include <iostream>
#include <string>

using std::vector;

int main(int argc, char* argv[]) {

        std::string model_file = "model.onnx";
        std::string word_symbol_file = "words.txt";
        std::string wav_audio = argv[1];
        InitRecognizer(model_file, word_symbol_file);

        AcceptWav(wav_audio);
        
        std::string result = GetResult();
        std::cout << "result :" << result << std::endl;
        Reset();
}
