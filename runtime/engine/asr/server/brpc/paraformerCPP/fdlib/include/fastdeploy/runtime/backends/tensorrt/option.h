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
#include <map>
#include <string>
#include <vector>

namespace fastdeploy {

/*! @brief Option object to configure TensorRT backend
 */
struct TrtBackendOption {
  /// `max_batch_size`, it's deprecated in TensorRT 8.x
  size_t max_batch_size = 32;

  /// `max_workspace_size` for TensorRT
  size_t max_workspace_size = 1 << 30;

  /// Enable log while converting onnx model to tensorrt
  bool enable_log_info = false;

  /*
   * @brief Enable half precison inference, on some device not support half precision, it will fallback to float32 mode
   */
  bool enable_fp16 = false;

  /** \brief Set shape range of input tensor for the model that contain dynamic input shape while using TensorRT backend
   *
   * \param[in] tensor_name The name of input for the model which is dynamic shape
   * \param[in] min The minimal shape for the input tensor
   * \param[in] opt The optimized shape for the input tensor, just set the most common shape, if set as default value, it will keep same with min_shape
   * \param[in] max The maximum shape for the input tensor, if set as default value, it will keep same with min_shape
   */
  void SetShape(const std::string& tensor_name,
                const std::vector<int32_t>& min,
                const std::vector<int32_t>& opt,
                const std::vector<int32_t>& max) {
    min_shape[tensor_name].clear();
    max_shape[tensor_name].clear();
    opt_shape[tensor_name].clear();
    min_shape[tensor_name].assign(min.begin(), min.end());
    if (opt.size() == 0) {
      opt_shape[tensor_name].assign(min.begin(), min.end());
    } else {
      opt_shape[tensor_name].assign(opt.begin(), opt.end());
    }
    if (max.size() == 0) {
      max_shape[tensor_name].assign(min.begin(), min.end());
    } else {
      max_shape[tensor_name].assign(max.begin(), max.end());
    }
  }
  /**
   * @brief Set cache file path while use TensorRT backend. Loadding a Paddle/ONNX model and initialize TensorRT will take a long time, by this interface it will save the tensorrt engine to `cache_file_path`, and load it directly while execute the code again
   */
  std::string serialize_file = "";

  // The below parameters may be removed in next version, please do not
  // visit or use them directly
  std::map<std::string, std::vector<int32_t>> max_shape;
  std::map<std::string, std::vector<int32_t>> min_shape;
  std::map<std::string, std::vector<int32_t>> opt_shape;
  bool enable_pinned_memory = false;
  void* external_stream_ = nullptr;
  int gpu_id = 0;
  std::string model_file = "";   // Path of model file
  std::string params_file = "";  // Path of parameters file, can be empty
  // format of input model
  ModelFormat model_format = ModelFormat::AUTOREC;
};


}  // namespace fastdeploy
