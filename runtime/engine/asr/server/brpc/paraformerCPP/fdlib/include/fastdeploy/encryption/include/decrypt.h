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

#include <stdio.h>
#include <string>

#include "fastdeploy/utils/utils.h"

#ifndef PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_DECRYPT_H
#define PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_DECRYPT_H
namespace fastdeploy {
#ifdef __cplusplus
extern "C" {
#endif

/** \brief check stream is encrypted or not
 *
 * \param[in] cipher_stream The encrypted stream
 * \return 0 if stream is encrypted.
 */
FASTDEPLOY_DECL int CheckStreamEncrypted(std::istream& cipher_stream);


/** \brief decrypt an encrypted stream
 *
 * \param[in] cipher_stream The encrypted stream
 * \param[in] plain_stream The decrypted stream
 * \param[in] key_base64 The key for decryption
 * \return 0 if decrypt success.
 */
FASTDEPLOY_DECL int DecryptStream(std::istream& cipher_stream,
                                 std::ostream& plain_stream,
                                 const std::string& key_base64);


/** \brief decrypt an encrypted string
 *
 * \param[in] cipher The encrypted string
 * \param[in] key The key for decryption
 * \return The decrypted string
 */
FASTDEPLOY_DECL std::string Decrypt(const std::string& cipher,
                  const std::string& key);
#ifdef __cplusplus
}
#endif
}  // namespace fastdeploy
#endif  // PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_DECRYPT_H
