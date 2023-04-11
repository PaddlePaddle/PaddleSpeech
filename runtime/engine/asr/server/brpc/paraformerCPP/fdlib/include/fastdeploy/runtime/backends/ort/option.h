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
namespace fastdeploy {

/*! @brief Option object to configure ONNX Runtime backend
 */
struct OrtBackendOption {
  /*
   * @brief Level of graph optimization, -1: mean default(Enable all the optimization strategy)/0: disable all the optimization strategy/1: enable basic strategy/2:enable extend strategy/99: enable all
   */
  int graph_optimization_level = -1;
  /*
   * @brief Number of threads to execute the operator, -1: default
   */
  int intra_op_num_threads = -1;
  /*
   * @brief Number of threads to execute the graph, -1: default. This parameter only will bring effects while the `OrtBackendOption::execution_mode` set to 1.
   */
  int inter_op_num_threads = -1;
  /*
   * @brief Execution mode for the graph, -1: default(Sequential mode)/0: Sequential mode, execute the operators in graph one by one. /1: Parallel mode, execute the operators in graph parallelly.
   */
  int execution_mode = -1;
  /// Inference device, OrtBackend supports CPU/GPU
  Device device = Device::CPU;
  /// Inference device id
  int device_id = 0;

  void* external_stream_ = nullptr;
};
}  // namespace fastdeploy
