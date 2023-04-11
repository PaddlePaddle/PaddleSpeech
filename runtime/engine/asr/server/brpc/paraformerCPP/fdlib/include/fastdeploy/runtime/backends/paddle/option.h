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
#include "fastdeploy/runtime/backends/tensorrt/option.h"


namespace fastdeploy {

/*! @brief Option object to configure GraphCore IPU
 */
struct IpuOption {
  /// IPU device id
  int ipu_device_num;
  /// the batch size in the graph, only work when graph has no batch shape info
  int ipu_micro_batch_size;
  /// enable pipelining
  bool ipu_enable_pipelining;
  /// the number of batches per run in pipelining
  int ipu_batches_per_step;
  /// enable fp16
  bool ipu_enable_fp16;
  /// the number of graph replication
  int ipu_replica_num;
  /// the available memory proportion for matmul/conv
  float ipu_available_memory_proportion;
  /// enable fp16 partial for matmul, only work with fp16
  bool ipu_enable_half_partial;
};

/*! @brief Option object to configure Paddle Inference backend
 */
struct PaddleBackendOption {
  /// Print log information while initialize Paddle Inference backend
  bool enable_log_info = false;
  /// Enable MKLDNN while inference on CPU
  bool enable_mkldnn = true;
  /// Use Paddle Inference + TensorRT to inference model on GPU
  bool enable_trt = false;

  /*
   * @brief IPU option, this will configure the IPU hardware, if inference model in IPU
   */
  IpuOption ipu_option;

  /// Collect shape for model while enable_trt is true
  bool collect_trt_shape = false;
  /// Cache input shape for mkldnn while the input data will change dynamiclly
  int mkldnn_cache_size = -1;
  /// initialize memory size(MB) for GPU
  int gpu_mem_init_size = 100;
  /// The option to enable fixed size optimization for transformer model
  bool enable_fixed_size_opt = false;

  /// Disable type of operators run on TensorRT
  void DisableTrtOps(const std::vector<std::string>& ops) {
    trt_disabled_ops_.insert(trt_disabled_ops_.end(), ops.begin(), ops.end());
  }

  /// Delete pass by name
  void DeletePass(const std::string& pass_name) {
    delete_pass_names.push_back(pass_name);
  }

  void SetIpuConfig(bool enable_fp16, int replica_num,
                                   float available_memory_proportion,
                                   bool enable_half_partial) {
    ipu_option.ipu_enable_fp16 = enable_fp16;
    ipu_option.ipu_replica_num = replica_num;
    ipu_option.ipu_available_memory_proportion =
        available_memory_proportion;
    ipu_option.ipu_enable_half_partial = enable_half_partial;
  }

  // The belowing parameters may be removed, please do not
  // read or write them directly
  TrtBackendOption trt_option;
  bool enable_pinned_memory = false;
  void* external_stream_ = nullptr;
  Device device = Device::CPU;
  int device_id = 0;
  std::vector<std::string> trt_disabled_ops_{};
  int cpu_thread_num = 8;
  std::vector<std::string> delete_pass_names = {};
  std::string model_file = "";   // Path of model file
  std::string params_file = "";  // Path of parameters file, can be empty

  // load model and paramters from memory
  bool model_from_memory_ = false;
};
}  // namespace fastdeploy
