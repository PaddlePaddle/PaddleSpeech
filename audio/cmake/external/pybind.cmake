include(FetchContent)
include(ExternalProject)

#the pybind11 is from:https://github.com/pybind/pybind11
# Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

FetchContent_Declare(
  pybind
  URL      https://paddleaudio.bj.bcebos.com/build/v2.10.0.zip
  URL_HASH SHA256=225df6e6dea7cea7c5754d4ed954e9ca7c43947b849b3795f87cb56437f1bd19
)
FetchContent_MakeAvailable(pybind)
include_directories(${pybind_SOURCE_DIR}/include)

