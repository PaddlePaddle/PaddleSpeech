cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Use compiler ID "AppleClang" instead of "Clang" for XCode.
# Not setting this sometimes makes XCode C compiler gets detected as "Clang",
# even when the C++ one is detected as "AppleClang".
cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0025 NEW)

# Suppress warning flags in default MSVC configuration.  It's not
# mandatory that we do this (and we don't if cmake is old), but it's
# nice when it's possible, and it's possible on our Windows configs.
if(NOT CMAKE_VERSION VERSION_LESS 3.15.0)
  cmake_policy(SET CMP0092 NEW)
endif()

project(paddlespeech)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "paddlespeech requires -std=c++14. Please remove -std=c++ settings in your environment.")
endif()


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)


# Options
option(BUILD_SOX "Build libsox statically" ON)
option(BUILD_MAD "Enable libmad" ON)
option(BUILD_PADDLEAUDIO_PYTHON_EXTENSION "Build Python extension" ON)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
message(STATUS "Python_INCLUDE_DIR=" ${Python_INCLUDE_DIR})

add_subdirectory(paddlespeech/audio/third_party)
add_subdirectory(paddlespeech/audio/src)

include(cmake/summary.cmake)
onnx_print_configuration_summary()