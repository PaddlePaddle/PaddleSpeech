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

#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "fastdeploy/utils/utils.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fastdeploy {

class BasePlugin : public nvinfer1::IPluginV2DynamicExt {
 protected:
  void setPluginNamespace(const char* libNamespace) noexcept override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return mNamespace.c_str();
  }

  std::string mNamespace;
};

class BaseCreator : public nvinfer1::IPluginCreator {
 public:
  void setPluginNamespace(const char* libNamespace) noexcept override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return mNamespace.c_str();
  }

 protected:
  std::string mNamespace;
};

typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

// Write values into buffer
template <typename T> void write(char*& buffer, const T& val) {
  std::memcpy(buffer, &val, sizeof(T));
  buffer += sizeof(T);
}

// Read values from buffer
template <typename T> T read(const char*& buffer) {
  T val{};
  std::memcpy(&val, buffer, sizeof(T));
  buffer += sizeof(T);
  return val;
}

}  // namespace fastdeploy
