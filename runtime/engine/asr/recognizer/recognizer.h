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

#pragma once

#include <string>
#include <vector>

bool InitRecognizer(const std::string& model_file, 
                    const std::string& word_symbol_table_file,
                    const std::string& fst_file,
                    int num_instance);
int GetRecognizerInstanceId();
void InitDecoder(int instance_id);
void AcceptData(const std::vector<float>& waves, int instance_id);
void SetInputFinished(int instance_id);
std::string GetFinalResult(int instance_id);