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

#include <string>
#include <vector>

#ifndef PADDLE_MODEL_PROTECT_SYSTEM_UTIL_H
#define PADDLE_MODEL_PROTECT_SYSTEM_UTIL_H
namespace fastdeploy {
namespace util {

class SystemUtils {
 public:
  static std::string random_key_iv(int len);
  static std::string random_str(int len);
  static int check_key_match(const char* key, const char* filepath);
  static int check_key_match(const std::string& key,
                             std::istream& cipher_stream);
  static int check_file_encrypted(const char* filepath);
  static int check_file_encrypted(std::istream& cipher_stream);
  static int check_pattern_exist(const std::vector<std::string>& vecs,
                                 const std::string& pattern);

 private:
  inline static int intN(int n);
};

}  // namespace util
}  // namespace fastdeploy
#endif  // PADDLE_MODEL_PROTECT_SYSTEM_UTIL_H
