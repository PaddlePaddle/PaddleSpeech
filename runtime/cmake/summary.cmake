# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(pps_summary)
  message(STATUS "")
  message(STATUS "*************PaddleSpeech Building Summary**********")
  message(STATUS "  CMake version             : ${CMAKE_VERSION}")
  message(STATUS "  CMake command             : ${CMAKE_COMMAND}")
  message(STATUS "  UNIX                      : ${UNIX}")
  message(STATUS "  ANDROID                   : ${ANDROID}")
  message(STATUS "  System                    : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler              : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version      : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  CXX flags                 : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  Build type                : ${CMAKE_BUILD_TYPE}")
  get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
  message(STATUS "  Compile definitions       : ${tmp}")
  message(STATUS "  CMAKE_PREFIX_PATH         : ${CMAKE_PREFIX_PATH}")
  message(STATUS "  CMAKE_INSTALL_PREFIX      : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "  CMAKE_MODULE_PATH         : ${CMAKE_MODULE_PATH}")
  message(STATUS "  CMAKE_SYSTEM_NAME         : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "")

  message(STATUS "  WITH_ASR                  : ${WITH_ASR}")
  message(STATUS "  WITH_CLS                  : ${WITH_CLS}")
  message(STATUS "  WITH_VAD                  : ${WITH_VAD}")
  message(STATUS "  WITH_GPU                  : ${WITH_GPU}")
  message(STATUS "  WITH_TESTING              : ${WITH_TESTING}")
  message(STATUS "  WITH_PROFILING            : ${WITH_PROFILING}")
  message(STATUS "  FASTDEPLOY_INSTALL_DIR    : ${FASTDEPLOY_INSTALL_DIR}")
  if(WITH_GPU)
    message(STATUS "  CUDA_DIRECTORY            : ${CUDA_DIRECTORY}")
  endif()

  if(ANDROID)
    message(STATUS "  ANDROID_ABI               : ${ANDROID_ABI}")
    message(STATUS "  ANDROID_PLATFORM          : ${ANDROID_PLATFORM}")
    message(STATUS "  ANDROID_NDK               : ${ANDROID_NDK}")
    message(STATUS "  ANDROID_NDK_VERSION       : ${CMAKE_ANDROID_NDK_VERSION}")
  endif() 
  if (WITH_ASR)
    message(STATUS "  Python executable         : ${PYTHON_EXECUTABLE}")
    message(STATUS "  Python includes           : ${PYTHON_INCLUDE_DIR}")
  endif()
endfunction()

pps_summary()