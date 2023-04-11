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

#ifndef PADDLE_MODEL_PROTECT_UTIL_CRYPTO_SHA256_UTILS_H
#define PADDLE_MODEL_PROTECT_UTIL_CRYPTO_SHA256_UTILS_H
namespace fastdeploy {
namespace util {
namespace crypto {

class SHA256Utils {
 public:
  static void sha256(const void* data, size_t len, unsigned char* md);
  static std::vector<unsigned char> sha256(const void* data, size_t len);
  static std::vector<unsigned char> sha256(
      const std::vector<unsigned char>& data);
  static std::string sha256_string(const void* data, size_t len);
  static std::string sha256_string(const std::vector<unsigned char>& data);
  static std::string sha256_string(const std::string& string);
  static std::string sha256_file(const std::string& path);
};

}  // namespace crypto
}  // namespace util
}  // namespace fastdeploy
#endif  // PADDLE_MODEL_PROTECT_UTIL_CRYPTO_SHA256_UTILS_H
