// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/core/fd_type.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fastdeploy {

/*! @brief Option object to configure Poros backend
 */
struct PorosBackendOption {
  Device device = Device::CPU;
  int device_id = 0;
  bool long_to_int = true;
  // There is calculation precision in tf32 mode on A10, it can bring some
  // performance improvement, but there may be diff
  bool use_nvidia_tf32 = false;
  // Threshold for the number of non-const ops
  int32_t unconst_ops_thres = -1;
  std::string poros_file = "";
  std::vector<FDDataType> prewarm_datatypes = {FDDataType::FP32};
  // TRT options
  bool enable_fp16 = false;
  bool enable_int8 = false;
  bool is_dynamic = false;
  size_t max_batch_size = 32;
  size_t max_workspace_size = 1 << 30;
};

}  // namespace fastdeploy
