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

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "fastdeploy/core/allocate.h"
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

struct FDInferDeleter {
  template <typename T> void operator()(T* obj) const {
    if (obj) {
      delete obj;
      //      obj->destroy();
    }
  }
};

template <typename T> using FDUniquePtr = std::unique_ptr<T, FDInferDeleter>;

int64_t Volume(const nvinfer1::Dims& d);

nvinfer1::Dims ToDims(const std::vector<int>& vec);
nvinfer1::Dims ToDims(const std::vector<int64_t>& vec);

size_t TrtDataTypeSize(const nvinfer1::DataType& dtype);

FDDataType GetFDDataType(const nvinfer1::DataType& dtype);

nvinfer1::DataType ReaderDtypeToTrtDtype(int reader_dtype);

FDDataType ReaderDtypeToFDDtype(int reader_dtype);

std::vector<int> ToVec(const nvinfer1::Dims& dim);

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
  out << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i != vec.size() - 1) {
      out << vec[i] << ", ";
    } else {
      out << vec[i] << "]";
    }
  }
  return out;
}

template <typename AllocFunc, typename FreeFunc> class FDGenericBuffer {
 public:
  //!
  //! \brief Construct an empty buffer.
  //!
  explicit FDGenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
      : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr),
        mExternal_buffer(nullptr) {}

  //!
  //! \brief Construct a buffer with the specified allocation size in bytes.
  //!
  FDGenericBuffer(size_t size, nvinfer1::DataType type)
      : mSize(size), mCapacity(size), mType(type) {
    if (!allocFn(&mBuffer, this->nbBytes())) {
      throw std::bad_alloc();
    }
  }

  //!
  //! \brief This use to skip memory copy step.
  //!
  FDGenericBuffer(size_t size, nvinfer1::DataType type, void* buffer)
      : mSize(size), mCapacity(size), mType(type) {
    mExternal_buffer = buffer;
  }

  FDGenericBuffer(FDGenericBuffer&& buf)
      : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType),
        mBuffer(buf.mBuffer) {
    buf.mSize = 0;
    buf.mCapacity = 0;
    buf.mType = nvinfer1::DataType::kFLOAT;
    buf.mBuffer = nullptr;
  }

  FDGenericBuffer& operator=(FDGenericBuffer&& buf) {
    if (this != &buf) {
      freeFn(mBuffer);
      mSize = buf.mSize;
      mCapacity = buf.mCapacity;
      mType = buf.mType;
      mBuffer = buf.mBuffer;
      // Reset buf.
      buf.mSize = 0;
      buf.mCapacity = 0;
      buf.mBuffer = nullptr;
    }
    return *this;
  }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  void* data() {
    if (mExternal_buffer != nullptr)
      return mExternal_buffer;
    return mBuffer;
  }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  const void* data() const {
    if (mExternal_buffer != nullptr)
      return mExternal_buffer;
    return mBuffer;
  }

  //!
  //! \brief Returns the size (in number of elements) of the buffer.
  //!
  size_t size() const { return mSize; }

  //!
  //! \brief Returns the size (in bytes) of the buffer.
  //!
  size_t nbBytes() const { return this->size() * TrtDataTypeSize(mType); }

  //!
  //! \brief Returns the dtype of the buffer.
  //!
  nvinfer1::DataType dtype() const { return mType; }

  //!
  //! \brief Set user memory buffer for TRT Buffer
  //!
  void SetExternalData(size_t size, nvinfer1::DataType type, void* buffer) {
    mSize = mCapacity = size;
    mType = type;
    mExternal_buffer = const_cast<void*>(buffer);
  }

  //!
  //! \brief Set user memory buffer for TRT Buffer
  //!
  void SetExternalData(const nvinfer1::Dims& dims, const void* buffer) {
    mSize = mCapacity = Volume(dims);
    mExternal_buffer = const_cast<void*>(buffer);
  }

  //!
  //! \brief Resizes the buffer. This is a no-op if the new size is smaller than
  //! or equal to the current capacity.
  //!
  void resize(size_t newSize) {
    mExternal_buffer = nullptr;
    mSize = newSize;
    if (mCapacity < newSize) {
      freeFn(mBuffer);
      if (!allocFn(&mBuffer, this->nbBytes())) {
        throw std::bad_alloc{};
      }
      mCapacity = newSize;
    }
  }

  //!
  //! \brief Overload of resize that accepts Dims
  //!
  void resize(const nvinfer1::Dims& dims) { return this->resize(Volume(dims)); }

  ~FDGenericBuffer() {
    mExternal_buffer = nullptr;
    freeFn(mBuffer);
  }

 private:
  size_t mSize{0}, mCapacity{0};
  nvinfer1::DataType mType;
  void* mBuffer;
  void* mExternal_buffer;
  AllocFunc allocFn;
  FreeFunc freeFn;
};

using FDDeviceBuffer = FDGenericBuffer<FDDeviceAllocator, FDDeviceFree>;
using FDDeviceHostBuffer =
    FDGenericBuffer<FDDeviceHostAllocator, FDDeviceHostFree>;

class FDTrtLogger : public nvinfer1::ILogger {
 public:
  static FDTrtLogger* logger;
  static FDTrtLogger* Get() {
    if (logger != nullptr) {
      return logger;
    }
    logger = new FDTrtLogger();
    return logger;
  }
  void SetLog(bool enable_info = false, bool enable_warning = false) {
    enable_info_ = enable_info;
    enable_warning_ = enable_warning;
  }

  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override {
    if (severity == nvinfer1::ILogger::Severity::kINFO) {
      if (enable_info_) {
        FDINFO << msg << std::endl;
      }
    } else if (severity == nvinfer1::ILogger::Severity::kWARNING) {
      if (enable_warning_) {
        FDWARNING << msg << std::endl;
      }
    } else if (severity == nvinfer1::ILogger::Severity::kERROR) {
      FDERROR << msg << std::endl;
    } else if (severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR) {
      FDASSERT(false, "%s", msg);
    }
  }

 private:
  bool enable_info_ = false;
  bool enable_warning_ = false;
};

struct ShapeRangeInfo {
  explicit ShapeRangeInfo(const std::vector<int64_t>& new_shape) {
    shape.assign(new_shape.begin(), new_shape.end());
    min.resize(new_shape.size());
    max.resize(new_shape.size());
    is_static.resize(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (new_shape[i] > 0) {
        min[i] = new_shape[i];
        max[i] = new_shape[i];
        is_static[i] = 1;
      } else {
        min[i] = -1;
        max[i] = -1;
        is_static[i] = 0;
      }
    }
  }

  std::string name;
  std::vector<int64_t> shape;
  std::vector<int64_t> min;
  std::vector<int64_t> max;
  std::vector<int64_t> opt;
  std::vector<int8_t> is_static;
  // return
  // -1: new shape is inillegal
  // 0 : new shape is able to inference
  // 1 : new shape is out of range, need to update engine
  int Update(const std::vector<int64_t>& new_shape);
  int Update(const std::vector<int>& new_shape) {
    std::vector<int64_t> new_shape_int64(new_shape.begin(), new_shape.end());
    return Update(new_shape_int64);
  }

  friend std::ostream& operator<<(std::ostream& out,
                                  const ShapeRangeInfo& info) {
    out << "Input name: " << info.name << ", shape=" << info.shape
        << ", min=" << info.min << ", max=" << info.max << std::endl;
    return out;
  }
};

}  // namespace fastdeploy
