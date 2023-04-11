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

#include <iostream>
#include <string>

#include "fastdeploy/utils/utils.h"

#ifndef PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_ENCRYPT_H
#define PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_ENCRYPT_H
namespace fastdeploy {
#ifdef __cplusplus
extern "C" {
#endif

/** \brief generate a random key(base64-32bytes) for an encrypted model
 *
 * \return std::string key
 */
FASTDEPLOY_DECL std::string GenerateRandomKey();

/** \brief encrypt a std::istream with key
 *
 * \param[in] keydata The key(base64-32bytes) for encryption
 * \param[in] in_stream The plain stream
 * \param[in] out_stream The ecrypted stream
 * \return true if encrypt successed, otherwise false
 */
FASTDEPLOY_DECL int EncryptStream(const std::string &keydata,
                                 std::istream& in_stream,
                                 std::ostream& out_stream);

/** \brief encrypt a string with key
 *
 * \param[in] input The input string for encryption
 * \param[in] key If not given by user, generate key automatically.
 * \return std::vector<std::string> [encrypted string, key]
 */
FASTDEPLOY_DECL std::vector<std::string> Encrypt(const std::string& input,
                  const std::string& key = GenerateRandomKey());

#ifdef __cplusplus
}
#endif
}  // namespace fastdeploy
#endif  // PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_ENCRYPT_H
