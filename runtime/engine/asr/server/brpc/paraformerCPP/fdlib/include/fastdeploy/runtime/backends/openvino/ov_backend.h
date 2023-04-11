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

#include "fastdeploy/runtime/backends/backend.h"
#include "fastdeploy/utils/unique_ptr.h"
#include "fastdeploy/runtime/backends/openvino/option.h"
#include "openvino/openvino.hpp"

namespace fastdeploy {

class OpenVINOBackend : public BaseBackend {
 public:
  static ov::Core core_;
  OpenVINOBackend() {}
  virtual ~OpenVINOBackend() = default;

  bool Init(const RuntimeOption& option);

  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs,
             bool copy_to_fd = true) override;

  int NumInputs() const override;

  int NumOutputs() const override;

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

  std::unique_ptr<BaseBackend> Clone(RuntimeOption &runtime_option,
                                     void* stream = nullptr,
                                     int device_id = -1) override;
  
 private:
  bool
  InitFromPaddle(const std::string& model_file, const std::string& params_file,
                 const OpenVINOBackendOption& option = OpenVINOBackendOption());

  bool
  InitFromOnnx(const std::string& model_file,
               const OpenVINOBackendOption& option = OpenVINOBackendOption());


  void InitTensorInfo(const std::vector<ov::Output<ov::Node>>& ov_outputs,
                      std::map<std::string, TensorInfo>* tensor_infos);

  ov::CompiledModel compiled_model_;
  ov::InferRequest request_;
  OpenVINOBackendOption option_;
  std::vector<TensorInfo> input_infos_;
  std::vector<TensorInfo> output_infos_;
};

}  // namespace fastdeploy