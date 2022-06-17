
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(paddleaudio VERSION 0.1)

# cmake dir
set(paddleaudio_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

# Modules
list(APPEND CMAKE_MODULE_PATH ${paddleaudio_cmake_dir}/external)
list(APPEND CMAKE_MODULE_PATH ${paddleaudio_cmake_dir})
include(FetchContent)
include(ExternalProject)

# fc_patch dir
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_patch "fc_patch" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_patch})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14 -pthread -fPIC -O0 -Wall -g")

# see the thirdparty/kaldi/base/kaldi-types.h
# compile kaldi without openfst
add_definitions("-DCOMPILE_WITHOUT_OPENFST")

include(openblas)
include(pybind)
#set(fc_patch /paddle/mnt/zhouyang/wks2/PaddleSpeech/paddleaudio/fc_patch)
#set(OpenBLAS_INSTALL_PREFIX ${fc_patch}/openblas-install)
#link_directories(${OpenBLAS_INSTALL_PREFIX}/lib)
#include_directories(${OpenBLAS_INSTALL_PREFIX}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kaldi)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include)
include_directories(/usr/include/python3.7m)
add_subdirectory(third_party)
add_subdirectory(kaldi_frontend)
