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
#include <map>
#include <set>
namespace fastdeploy {

/*! @brief Option object to configure OpenVINO backend
 */
struct OpenVINOBackendOption {
  std::string device = "CPU";
  int cpu_thread_num = -1;

  /// Number of streams while use OpenVINO
  int num_streams = 0;

  /**
   * @brief Set device name for OpenVINO, default 'CPU', can also be 'AUTO', 'GPU', 'GPU.1'....
   */
  void SetDevice(const std::string& name = "CPU") {
    device = name;
  }

  /**
   * @brief Set shape info for OpenVINO
   */
  void SetShapeInfo(
      const std::map<std::string, std::vector<int64_t>>& _shape_infos) {
    shape_infos = _shape_infos;
  }

  /**
   * @brief While use OpenVINO backend with intel GPU, use this interface to specify operators run on CPU
   */
  void SetCpuOperators(const std::vector<std::string>& operators) {
    for (const auto& op : operators) {
      cpu_operators.insert(op);
    }
  }

  std::map<std::string, std::vector<int64_t>> shape_infos;
  std::set<std::string> cpu_operators{"MulticlassNms"};
};
}  // namespace fastdeploy
