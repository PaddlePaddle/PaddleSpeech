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
#include "fastdeploy/runtime/backends/paddle/option.h"
#ifdef ENABLE_PADDLE2ONNX
#include "paddle2onnx/converter.h"
#endif
#include "fastdeploy/utils/unique_ptr.h"
#include "paddle_inference_api.h"  // NOLINT

namespace fastdeploy {

// convert FD device to paddle place type
paddle_infer::PlaceType ConvertFDDeviceToPlace(Device device);

// Share memory buffer with paddle_infer::Tensor from fastdeploy::FDTensor
void ShareTensorFromFDTensor(paddle_infer::Tensor* tensor, FDTensor& fd_tensor);

void ShareOutTensorFromFDTensor(paddle_infer::Tensor* tensor,
                             FDTensor& fd_tensor);

// convert paddle_infer::Tensor to fastdeploy::FDTensor
// if copy_to_fd is true, copy memory data to FDTensor
/// else share memory to FDTensor
void PaddleTensorToFDTensor(std::unique_ptr<paddle_infer::Tensor>& tensor,
                            FDTensor* fd_tensor, bool copy_to_fd);

// Convert data type from paddle inference to fastdeploy
FDDataType PaddleDataTypeToFD(const paddle_infer::DataType& dtype);

// Convert data type from paddle2onnx::PaddleReader to fastdeploy
FDDataType ReaderDataTypeToFD(int32_t dtype);

class PaddleBackend : public BaseBackend {
 public:
  PaddleBackend() {}
  virtual ~PaddleBackend() = default;
  bool Init(const RuntimeOption& option);
  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs,
             bool copy_to_fd = true) override;

  int NumInputs() const override { return inputs_desc_.size(); }

  int NumOutputs() const override { return outputs_desc_.size(); }

  std::unique_ptr<BaseBackend> Clone(RuntimeOption &runtime_option,
                                     void* stream = nullptr,
                                     int device_id = -1) override;

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

 private:
  void BuildOption(const PaddleBackendOption& option);

  bool InitFromPaddle(const std::string& model_buffer,
                     const std::string& params_buffer,
                     const PaddleBackendOption& option = PaddleBackendOption());

  void
  CollectShapeRun(paddle_infer::Predictor* predictor,
                  const std::map<std::string, std::vector<int>>& shape) const;
  void GetDynamicShapeFromOption(
      const PaddleBackendOption& option,
      std::map<std::string, std::vector<int>>* max_shape,
      std::map<std::string, std::vector<int>>* min_shape,
      std::map<std::string, std::vector<int>>* opt_shape) const;
  void SetTRTDynamicShapeToConfig(const PaddleBackendOption& option);
  PaddleBackendOption option_;
  paddle_infer::Config config_;
  std::shared_ptr<paddle_infer::Predictor> predictor_;
  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
};
}  // namespace fastdeploy
