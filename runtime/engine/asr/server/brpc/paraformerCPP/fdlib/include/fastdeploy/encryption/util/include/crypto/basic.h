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

#ifndef PADDLE_MODEL_PROTECT_UTIL_BASIC_H
#define PADDLE_MODEL_PROTECT_UTIL_BASIC_H

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <sstream>
namespace fastdeploy {
namespace util {
namespace crypto {

class Basic {
 public:
  /**
   * \brief        byte to hex
   *
   * \note         byte to hex.
   *
   *
   * \param in_byte  byte array(in)
   * \param len      byte array length(in)
   * \param out_hex  the hex string(in)
   *
   *
   * \return         return  0 if successful
   */
  static int byte_to_hex(const unsigned char* in_byte, int len,
                         std::string& out_hex);  // NOLINT

  /**
   * \brief        hex to byte
   *
   * \note         hex to byte.
   *
   *
   * \param in_hex    the hex string(in)
   * \param out_byte  byte array(out)
   *
   * \return         return  0 if successful
   *                        -1 invalid in_hex
   */
  static int hex_to_byte(const std::string& in_hex, unsigned char* out_byte);

  /**
   * \brief        get random char for length
   *
   * \note         get random char for length
   *
   *
   * \param array     to be random(out)
   * \param len       array length(in)
   *
   * \return         return  0 if successful
   *                        -1 invalid parameters
   */
  static int random(unsigned char* random, int len);
};

}  // namespace crypto
}  // namespace util
}  // namespace fastdeploy
#endif  // PADDLE_MODEL_PROTECT_UTIL_BASIC_H
