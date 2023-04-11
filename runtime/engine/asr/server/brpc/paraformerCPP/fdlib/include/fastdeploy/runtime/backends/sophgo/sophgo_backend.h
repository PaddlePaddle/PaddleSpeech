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

#include "fastdeploy/runtime/backends/backend.h"
#include "fastdeploy/core/fd_tensor.h"
#include "bmruntime_interface.h" // NOLINT
#include "bmlib_runtime.h" // NOLINT
#include "fastdeploy/runtime/backends/sophgo/option.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fastdeploy {

class SophgoBackend : public BaseBackend {
 public:
  SophgoBackend() = default;
  virtual ~SophgoBackend();
  bool Init(const RuntimeOption& option);

  int NumInputs() const override {
      return static_cast<int>(inputs_desc_.size());
  }

  int NumOutputs() const override {
      return static_cast<int>(outputs_desc_.size());
  }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;
  bool Infer(std::vector<FDTensor>& inputs,
              std::vector<FDTensor>* outputs,
              bool copy_to_fd = true) override;

 private:
  bool LoadModel(void* model);
  bool GetSDKAndDeviceVersion();
  bool GetModelInputOutputInfos();

  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
  std::string net_name_;

  bm_handle_t handle_;
  void * p_bmrt_ = nullptr;

  bool infer_init = false;

  const bm_net_info_t* net_info_ = nullptr;

  // SophgoTPU2BackendOption option_;

  static FDDataType SophgoTensorTypeToFDDataType(bm_data_type_t type);
  static bm_data_type_t FDDataTypeToSophgoTensorType(FDDataType type);
};
}  // namespace fastdeploy
