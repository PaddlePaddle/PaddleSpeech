# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 ZeXuan Li (lizexuan@huya.com)
#                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
#                    hamddct@gmail.com (Mddct)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(WITH_ONNX)
  set(ONNX_VERSION "1.12.0")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-win-x64-${ONNX_VERSION}.zip")
    set(URL_HASH "SHA256=8b5d61204989350b7904ac277f5fbccd3e6736ddbb6ec001e412723d71c9c176")
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz")
      set(URL_HASH "SHA256=5820d9f343df73c63b6b2b174a1ff62575032e171c9564bcf92060f46827d0ac")
    else()
      set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
      set(URL_HASH "SHA256=5d503ce8540358b59be26c675e42081be14a3e833a5301926f555451046929c5")
    endif()
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz")
    set(URL_HASH "SHA256=09b17f712f8c6f19bb63da35d508815b443cbb473e16c6192abfaa297c02f600")
  else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
  endif()

  FetchContent_Declare(onnxruntime
    URL ${ONNX_URL}
    URL_HASH ${URL_HASH}
  )
  FetchContent_MakeAvailable(onnxruntime)
  include_directories(${onnxruntime_SOURCE_DIR}/include)
  link_directories(${onnxruntime_SOURCE_DIR}/lib)

  if(MSVC)
    file(GLOB ONNX_DLLS "${onnxruntime_SOURCE_DIR}/lib/*.dll")
    file(COPY ${ONNX_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
  endif()

  add_definitions(-DUSE_ONNX)
endif()
