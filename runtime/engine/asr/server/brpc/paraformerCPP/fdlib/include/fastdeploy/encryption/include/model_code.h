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

#ifndef DYGRAPH_DEPLOY_CPP_ENCRYPTION_INCLUDE_MODEL_CODE_H_
#define DYGRAPH_DEPLOY_CPP_ENCRYPTION_INCLUDE_MODEL_CODE_H_
namespace fastdeploy {
#ifdef __cplusplus
extern  "C" {
#endif

enum  {
    CODE_OK                         = 0,
    CODE_OPEN_FAILED                = 100,
    CODE_READ_FILE_PTR_IS_NULL      = 101,
    CODE_AES_GCM_ENCRYPT_FIALED     = 102,
    CODE_AES_GCM_DECRYPT_FIALED     = 103,
    CODE_KEY_NOT_MATCH              = 104,
    CODE_KEY_LENGTH_ABNORMAL        = 105,
    CODE_NOT_EXIST_DIR              = 106,
    CODE_FILES_EMPTY_WITH_DIR       = 107,
    CODE_MODEL_FILE_NOT_EXIST       = 108,
    CODE_PARAMS_FILE_NOT_EXIST      = 109,
    CODE_MODEL_YML_FILE_NOT_EXIST   = 110,
    CODE_MKDIR_FAILED               = 111
};

#ifdef __cplusplus
}
#endif
}  // namespace fastdeploy
#endif  // DYGRAPH_DEPLOY_CPP_ENCRYPTION_INCLUDE_MODEL_CODE_H_
