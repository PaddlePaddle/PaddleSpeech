cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(paddleaudio VERSION 0.1)

string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -O0 -Wall -g")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SOX "Build libsox statically" ON)


# checkout the thirdparty/kaldi/base/kaldi-types.h
# compile kaldi without openfst
add_definitions("-DCOMPILE_WITHOUT_OPENFST")

include(openblas)
# include(pybind)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kaldi)
include_directories(/usr/include/python3.7m)

add_subdirectory(third_party)
add_subdirectory(src)
