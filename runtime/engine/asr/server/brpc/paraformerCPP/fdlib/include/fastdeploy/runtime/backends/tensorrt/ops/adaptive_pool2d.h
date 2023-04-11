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
#include "common.h"  // NOLINT
#include "fastdeploy/runtime/backends/common/cuda/adaptive_pool2d_kernel.h"

namespace fastdeploy {

class AdaptivePool2d : public BasePlugin {
 public:
  AdaptivePool2d(std::vector<int32_t> output_size, std::string pooling_type);

  AdaptivePool2d(const void* buffer, size_t length);

  ~AdaptivePool2d() override = default;

  int getNbOutputs() const noexcept override;

  nvinfer1::DimsExprs
  getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                      int nbInputs,
                      nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputType,
                                       int nbInputs) const noexcept override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) noexcept override;

  int initialize() noexcept override;

  void terminate() noexcept override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void* buffer) const noexcept override;

  const char* getPluginType() const noexcept override;

  const char* getPluginVersion() const noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) noexcept override;
  void destroy() noexcept override;

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

 private:
  std::vector<int32_t> output_size_;
  std::string pooling_type_;
};

class AdaptivePool2dPluginCreator : public BaseCreator {
 public:
  AdaptivePool2dPluginCreator();

  ~AdaptivePool2dPluginCreator() override = default;

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt*
  createPlugin(const char* name,
               const nvinfer1::PluginFieldCollection* fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData,
                    size_t serialLength) noexcept override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::vector<int32_t> output_size_;
  std::string pooling_type_;
};

REGISTER_TENSORRT_PLUGIN(AdaptivePool2dPluginCreator);

}  // namespace fastdeploy
