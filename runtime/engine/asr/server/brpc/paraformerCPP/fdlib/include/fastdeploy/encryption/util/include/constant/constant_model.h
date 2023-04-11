//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#ifndef PADDLE_MODEL_PROTECT_CONSTANT_CONSTANT_MODEL_H
#define PADDLE_MODEL_PROTECT_CONSTANT_CONSTANT_MODEL_H
namespace fastdeploy {
namespace constant {

const static std::string MAGIC_NUMBER = "PADDLE";  // NOLINT
const static std::string VERSION = "1";            // NOLINT

const static int MAGIC_NUMBER_LEN = 6;  // NOLINT
const static int VERSION_LEN = 1;       // NOLINT
const static int TAG_LEN = 128;         // NOLINT

}  // namespace constant
}  // namespace fastdeploy
#endif  // PADDLE_MODEL_PROTECT_CONSTANT_CONSTANT_MODEL_H
