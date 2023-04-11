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

#include <cuda_runtime_api.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "fastdeploy/runtime/backends/backend.h"
#include "fastdeploy/runtime/backends/tensorrt/utils.h"
#include "fastdeploy/runtime/backends/tensorrt/option.h"
#include "fastdeploy/utils/unique_ptr.h"

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  explicit Int8EntropyCalibrator2(const std::string& calibration_cache)
      : calibration_cache_(calibration_cache) {}

  int getBatchSize() const noexcept override { return 0; }

  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) noexcept override {
    return false;
  }

  const void* readCalibrationCache(size_t& length) noexcept override {
    length = calibration_cache_.size();
    return length ? calibration_cache_.data() : nullptr;
  }

  void writeCalibrationCache(const void* cache,
                             size_t length) noexcept override {
    fastdeploy::FDERROR << "NOT IMPLEMENT." << std::endl;
  }

 private:
  const std::string calibration_cache_;
};

namespace fastdeploy {

struct TrtValueInfo {
  std::string name;
  std::vector<int> shape;
  nvinfer1::DataType dtype;   // dtype of TRT model
  FDDataType original_dtype;  // dtype of original ONNX/Paddle model
};

std::vector<int> toVec(const nvinfer1::Dims& dim);
size_t TrtDataTypeSize(const nvinfer1::DataType& dtype);
FDDataType GetFDDataType(const nvinfer1::DataType& dtype);

class TrtBackend : public BaseBackend {
 public:
  TrtBackend() : engine_(nullptr), context_(nullptr) {}

  bool Init(const RuntimeOption& runtime_option);
  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs,
             bool copy_to_fd = true) override;

  int NumInputs() const { return inputs_desc_.size(); }
  int NumOutputs() const { return outputs_desc_.size(); }
  TensorInfo GetInputInfo(int index);
  TensorInfo GetOutputInfo(int index);
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;
  std::unique_ptr<BaseBackend> Clone(RuntimeOption &runtime_option,
                                     void* stream = nullptr,
                                     int device_id = -1) override;

  ~TrtBackend() {
    if (parser_) {
      parser_.reset();
    }
  }

 private:
  void BuildOption(const TrtBackendOption& option);

  bool InitFromPaddle(const std::string& model_buffer,
                      const std::string& params_buffer,
                      const TrtBackendOption& option = TrtBackendOption(),
                      bool verbose = false);
  bool InitFromOnnx(const std::string& model_buffer,
                    const TrtBackendOption& option = TrtBackendOption());

  TrtBackendOption option_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  FDUniquePtr<nvonnxparser::IParser> parser_;
  FDUniquePtr<nvinfer1::IBuilder> builder_;
  FDUniquePtr<nvinfer1::INetworkDefinition> network_;
  cudaStream_t stream_{};
  std::vector<void*> bindings_;
  std::vector<TrtValueInfo> inputs_desc_;
  std::vector<TrtValueInfo> outputs_desc_;
  std::map<std::string, FDDeviceBuffer> inputs_device_buffer_;
  std::map<std::string, FDDeviceBuffer> outputs_device_buffer_;
  std::map<std::string, int> io_name_index_;

  std::string calibration_str_;
  bool save_external_ = false;
  std::string model_file_name_ = "";

  // Sometimes while the number of outputs > 1
  // the output order of tensorrt may not be same
  // with the original onnx model
  // So this parameter will record to origin outputs
  // order, to help recover the rigt order
  std::map<std::string, int> outputs_order_;

  // temporary store onnx model content
  // once it used to build trt egnine done
  // it will be released
  std::string onnx_model_buffer_;
  // Stores shape information of the loaded model
  // For dynmaic shape will record its range information
  // Also will update the range information while inferencing
  std::map<std::string, ShapeRangeInfo> shape_range_info_;

  // If the final output tensor's dtype is different from the
  // model output tensor's dtype, then we need cast the data
  // to the final output's dtype.
  // E.g. When trt model output tensor is int32, but final tensor is int64
  // This map stores the casted tensors.
  std::map<std::string, FDTensor> casted_output_tensors_;

  void GetInputOutputInfo();
  bool CreateTrtEngineFromOnnx(const std::string& onnx_model_buffer);
  bool BuildTrtEngine();
  bool LoadTrtCache(const std::string& trt_engine_file);
  int ShapeRangeInfoUpdated(const std::vector<FDTensor>& inputs);
  void SetInputs(const std::vector<FDTensor>& inputs);
  void AllocateOutputsBuffer(std::vector<FDTensor>* outputs,
                             bool copy_to_fd = true);
};

}  // namespace fastdeploy
