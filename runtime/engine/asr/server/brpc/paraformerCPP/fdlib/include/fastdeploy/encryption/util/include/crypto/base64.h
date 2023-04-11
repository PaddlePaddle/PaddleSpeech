// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include <string>

#ifndef PADDLE_MODEL_PROTECT_UTIL_CRYPTO_BASE64_UTILS_H
#define PADDLE_MODEL_PROTECT_UTIL_CRYPTO_BASE64_UTILS_H
namespace fastdeploy {
namespace baidu {
namespace base {
namespace base64 {

std::string base64_encode(const std::string& input);
std::string base64_decode(const std::string& input);

}  // namespace base64
}  // namespace base
}  // namespace baidu
}  // namespace fastdeploy
#endif  // PADDLE_MODEL_PROTECT_BASE64_UTILS_H
