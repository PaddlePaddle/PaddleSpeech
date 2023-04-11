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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/core/fd_type.h"
#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy/benchmark/benchmark.h"

namespace fastdeploy {

/*! @brief Information of Tensor
 */
struct TensorInfo {
  std::string name;        ///< Name of tensor
  std::vector<int> shape;  ///< Shape of tensor
  FDDataType dtype;        ///< Data type of tensor

  friend std::ostream& operator<<(std::ostream& output,
                                  const TensorInfo& info) {
    output << "TensorInfo(name: " << info.name << ", shape: [";
    for (size_t i = 0; i < info.shape.size(); ++i) {
      if (i == info.shape.size() - 1) {
        output << info.shape[i];
      } else {
        output << info.shape[i] << ", ";
      }
    }
    output << "], dtype: " << Str(info.dtype) << ")";
    return output;
  }
};

class BaseBackend {
 public:
  bool initialized_ = false;

  BaseBackend() {}
  virtual ~BaseBackend() = default;

  virtual bool Initialized() const { return initialized_; }

  virtual bool Init(const RuntimeOption& option) {
    FDERROR << "Not Implement for "
            << option.backend << " in "
            << option.device << "."
            << std::endl;
    return false;
  }

  // Get number of inputs of the model
  virtual int NumInputs() const = 0;
  // Get number of outputs of the model
  virtual int NumOutputs() const = 0;
  // Get information of input tensor
  virtual TensorInfo GetInputInfo(int index) = 0;
  // Get information of output tensor
  virtual TensorInfo GetOutputInfo(int index) = 0;
  // Get information of all the input tensors
  virtual std::vector<TensorInfo> GetInputInfos() = 0;
  // Get information of all the output tensors
  virtual std::vector<TensorInfo> GetOutputInfos() = 0;

  // if copy_to_fd is true, copy memory data to FDTensor
  // else share memory to FDTensor(only Paddle、ORT、TRT、OpenVINO support it)
  virtual bool Infer(std::vector<FDTensor>& inputs,
                     std::vector<FDTensor>* outputs,
                     bool copy_to_fd = true) = 0;
  // Optional: For those backends which can share memory
  // while creating multiple inference engines with same model file
  virtual std::unique_ptr<BaseBackend> Clone(RuntimeOption &runtime_option,
                                             void *stream = nullptr,
                                             int device_id = -1) {
    FDERROR << "Clone no support" << std::endl;
    return nullptr;
  }

  benchmark::BenchmarkOption benchmark_option_;
  benchmark::BenchmarkResult benchmark_result_;
};

/** \brief Macros for Runtime benchmark profiling.
 * The param 'base_loop' for 'RUNTIME_PROFILE_LOOP_BEGIN'
 * indicates that the least number of times the loop
 * will repeat when profiling mode is not enabled.
 * In most cases, the value should be 1, i.e., results are
 * obtained by running the inference process once, when
 * the profile mode is turned off, such as ONNX Runtime,
 * OpenVINO, TensorRT, Paddle Inference, Paddle Lite,
 * RKNPU2, SOPHGO etc.
 *
 * example code @code
 * // OpenVINOBackend::Infer
 * RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
 * // do something ....
 * RUNTIME_PROFILE_LOOP_BEGIN(1)
 * // The codes which wrapped by 'BEGIN(1) ~ END' scope
 * // will only run once when profiling mode is not enabled.
 * request_.infer();
 * RUNTIME_PROFILE_LOOP_END
 * // do something ....
 * RUNTIME_PROFILE_LOOP_H2D_D2H_END
 *
 * @endcode In this case, No global variables inside a function
 * are wrapped by BEGIN and END, which may be required for
 * subsequent tasks. But, some times we need to set 'base_loop'
 * as 0, such as POROS.
 *
 * * example code @code
 * // PorosBackend::Infer
 * RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
 * // do something ....
 * RUNTIME_PROFILE_LOOP_BEGIN(0) // set 'base_loop' as 0
 * // The codes which wrapped by 'BEGIN(0) ~ END' scope
 * // will not run when profiling mode is not enabled.
 * auto poros_outputs = _poros_module->forward(poros_inputs);
 * RUNTIME_PROFILE_LOOP_END
 * // Run another inference beyond the scope of 'BEGIN ~ END'
 * // to get valid outputs for subsequent tasks.
 * auto poros_outputs = _poros_module->forward(poros_inputs);
 * // do something .... will use 'poros_outputs' ...
 * if (poros_outputs.isTensor()) {
 * // ...
 * }
 * RUNTIME_PROFILE_LOOP_H2D_D2H_END
 *
 * @endcode In this case, 'poros_outputs' inside a function
 * are wrapped by BEGIN and END, which may be required for
 * subsequent tasks. So, we set 'base_loop' as 0 and lanuch
 * another infer to get the valid outputs beyond the scope
 * of 'BEGIN ~ END' for subsequent tasks.
 */

#define RUNTIME_PROFILE_LOOP_BEGIN(base_loop)            \
  __RUNTIME_PROFILE_LOOP_BEGIN(benchmark_option_, (base_loop))
#define RUNTIME_PROFILE_LOOP_END                         \
  __RUNTIME_PROFILE_LOOP_END(benchmark_result_)
#define RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN               \
  __RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN(benchmark_option_, 1)
#define RUNTIME_PROFILE_LOOP_H2D_D2H_END                 \
  __RUNTIME_PROFILE_LOOP_H2D_D2H_END(benchmark_result_)

}  // namespace fastdeploy
