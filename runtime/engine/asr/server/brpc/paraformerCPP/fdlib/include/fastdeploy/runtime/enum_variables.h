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

/*! \file enum_variables.h
    \brief A brief file description.

    More details
 */

#pragma once
#include "fastdeploy/utils/utils.h"
#include <ostream>
#include <map>

namespace fastdeploy {


/*! Inference backend supported in FastDeploy */
enum Backend {
  UNKNOWN,  ///< Unknown inference backend
  ORT,  //< ONNX Runtime, support Paddle/ONNX format model,
  //< CPU/ Nvidia GPU DirectML
  TRT,  ///< TensorRT, support Paddle/ONNX format model, Nvidia GPU only
  PDINFER,  ///< Paddle Inference, support Paddle format model, CPU / Nvidia GPU
  POROS,    ///< Poros, support TorchScript format model, CPU / Nvidia GPU
  OPENVINO,   ///< Intel OpenVINO, support Paddle/ONNX format, CPU only
  LITE,       ///< Paddle Lite, support Paddle format model, ARM CPU only
  RKNPU2,     ///< RKNPU2, support RKNN format model, Rockchip NPU only
  SOPHGOTPU,  ///< SOPHGOTPU, support SOPHGO format model, Sophgo TPU only
};

/**
 * @brief Get all the available inference backend in FastDeploy
 */
FASTDEPLOY_DECL std::vector<Backend> GetAvailableBackends();

/**
 * @brief Check if the inference backend available
 */
FASTDEPLOY_DECL bool IsBackendAvailable(const Backend& backend);


enum FASTDEPLOY_DECL Device {
  CPU,
  GPU,
  RKNPU,
  IPU,
  TIMVX,
  KUNLUNXIN,
  ASCEND,
  SOPHGOTPUD,
  DIRECTML
};

/*! Deep learning model format */
enum ModelFormat {
  AUTOREC,      ///< Auto recognize the model format by model file name
  PADDLE,       ///< Model with paddlepaddle format
  ONNX,         ///< Model with ONNX format
  RKNN,         ///< Model with RKNN format
  TORCHSCRIPT,  ///< Model with TorchScript format
  SOPHGO,       ///< Model with SOPHGO format
};

/// Describle all the supported backends for specified model format
static std::map<ModelFormat, std::vector<Backend>>
    s_default_backends_by_format = {
  {ModelFormat::PADDLE, {Backend::PDINFER, Backend::LITE,
                      Backend::ORT, Backend::OPENVINO, Backend::TRT}},
  {ModelFormat::ONNX, {Backend::ORT, Backend::OPENVINO, Backend::TRT}},
  {ModelFormat::RKNN, {Backend::RKNPU2}},
  {ModelFormat::TORCHSCRIPT, {Backend::POROS}},
  {ModelFormat::SOPHGO, {Backend::SOPHGOTPU}}
};

/// Describle all the supported backends for specified device
static std::map<Device, std::vector<Backend>>
    s_default_backends_by_device = {
  {Device::CPU, {Backend::LITE, Backend::PDINFER, Backend::ORT,
                Backend::OPENVINO, Backend::POROS}},
  {Device::GPU, {Backend::PDINFER, Backend::ORT, Backend::TRT, Backend::POROS}},
  {Device::RKNPU, {Backend::RKNPU2}},
  {Device::IPU, {Backend::PDINFER}},
  {Device::TIMVX, {Backend::LITE}},
  {Device::KUNLUNXIN, {Backend::LITE}},
  {Device::ASCEND, {Backend::LITE}},
  {Device::SOPHGOTPUD, {Backend::SOPHGOTPU}},
  {Device::DIRECTML, {Backend::ORT}}
};

inline bool Supported(ModelFormat format, Backend backend) {
  auto iter = s_default_backends_by_format.find(format);
  if (iter == s_default_backends_by_format.end()) {
    FDERROR << "Didn't find format is registered in " <<
            "s_default_backends_by_format." << std::endl;
    return false;
  }
  for (size_t i = 0; i < iter->second.size(); ++i) {
    if (iter->second[i] == backend) {
      return true;
    }
  }
  std::string msg = Str(iter->second);
  FDERROR << backend << " only supports " << msg << ", but now it's "
                      << format << "." << std::endl;
  return false;
}

inline bool Supported(Device device, Backend backend) {
  auto iter = s_default_backends_by_device.find(device);
  if (iter == s_default_backends_by_device.end()) {
    FDERROR << "Didn't find device is registered in " <<
              "s_default_backends_by_device." << std::endl;
    return false;
  }
  for (size_t i = 0; i < iter->second.size(); ++i) {
    if (iter->second[i] == backend) {
      return true;
    }
  }
  std::string msg = Str(iter->second);
  FDERROR << backend << " only supports " << msg << ", but now it's "
          << device << "." << std::endl;
  return false;
}

FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& o, const Backend& b);
FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& o, const Device& d);
FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& o, const ModelFormat& f);
}  // namespace fastdeploy
