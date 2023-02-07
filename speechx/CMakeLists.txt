cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableCMP0048.cmake")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(system)

# Ninja Generator will set CMAKE_BUILD_TYPE to Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

project(paddlespeech VERSION 0.1)

set(CMAKE_VERBOSE_MAKEFILE on)

# set std-14
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
include(ExternalProject)

# fc_patch dir
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_patch "fc_patch" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_patch})

# compiler option
# Keep the same with openfst, -fPIC or -fpic
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14 -pthread -fPIC -O0 -Wall -g -ldl")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} --std=c++14 -pthread -fPIC -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} --std=c++14 -pthread -fPIC -O3 -Wall")

###############################################################################
# Option Configurations
###############################################################################
option(TEST_DEBUG "option for debug" OFF)
option(USE_PROFILING "enable c++ profling" OFF)
option(WITH_TESTING "unit test" ON)

option(USING_GPU "u2 compute on GPU." OFF)

###############################################################################
# Include third party
###############################################################################
include(gflags)

include(glog)

# openfst
include(openfst)
add_dependencies(openfst gflags glog)

# paddle lib
include(paddleinference)

# gtest
if(WITH_TESTING)
    include(gtest) # download, build, install gtest
endif()

# python/pybind11/threads
find_package(Threads REQUIRED)
# https://cmake.org/cmake/help/latest/module/FindPython3.html#module:FindPython3
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

if(Python3_FOUND)
  message(STATUS "Python3_FOUND = ${Python3_FOUND}")
  message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
  message(STATUS "Python3_LIBRARIES = ${Python3_LIBRARIES}")
  message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
  message(STATUS "Python3_LINK_OPTIONS = ${Python3_LINK_OPTIONS}")
  set(PYTHON_LIBRARIES ${Python3_LIBRARIES} CACHE STRING "python lib" FORCE)
  set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS} CACHE STRING "python inc" FORCE)
endif()

message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_INCLUDE_DIR = ${PYTHON_INCLUDE_DIR}")

if(pybind11_FOUND)
  message(STATUS "pybind11_INCLUDES = ${pybind11_INCLUDE_DIRS}")
  message(STATUS "pybind11_LIBRARIES=${pybind11_LIBRARIES}")
  message(STATUS "pybind11_DEFINITIONS=${pybind11_DEFINITIONS}")
endif()


# paddle libpaddle.so
# paddle include and link option
# -L/workspace/DeepSpeech-2.x/speechx/venv/lib/python3.7/site-packages/paddle/libs -L/workspace/DeepSpeech-2.x/speechx/venv/lib/python3.7/site-packages/paddle/fluid -l:libpaddle.so -l:libdnnl.so.2 -l:libiomp5.so
execute_process(
    COMMAND python -c "\
import os;\
import paddle;\
include_dir=paddle.sysconfig.get_include();\
paddle_dir=os.path.split(include_dir)[0];\
libs_dir=os.path.join(paddle_dir, 'libs');\
fluid_dir=os.path.join(paddle_dir, 'fluid');\
out=' '.join([\"-L\" + libs_dir, \"-L\" + fluid_dir]);\
out += \" -l:libpaddle.so -l:libdnnl.so.2 -l:libiomp5.so\"; print(out);\
    "
    OUTPUT_VARIABLE PADDLE_LINK_FLAGS
    RESULT_VARIABLE SUCESS)

message(STATUS PADDLE_LINK_FLAGS= ${PADDLE_LINK_FLAGS})
string(STRIP ${PADDLE_LINK_FLAGS} PADDLE_LINK_FLAGS)

# paddle compile option
# -I/workspace/DeepSpeech-2.x/speechx/venv/lib/python3.7/site-packages/paddle/include
execute_process(
    COMMAND python -c "\
import paddle; \
include_dir = paddle.sysconfig.get_include(); \
print(f\"-I{include_dir}\"); \
    "
    OUTPUT_VARIABLE PADDLE_COMPILE_FLAGS)
message(STATUS PADDLE_COMPILE_FLAGS= ${PADDLE_COMPILE_FLAGS})
string(STRIP ${PADDLE_COMPILE_FLAGS} PADDLE_COMPILE_FLAGS)


# for LD_LIBRARY_PATH
# set(PADDLE_LIB_DIRS /workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/libs/)
execute_process(
    COMMAND python -c "\
import os; \
import paddle; \
include_dir=paddle.sysconfig.get_include(); \
paddle_dir=os.path.split(include_dir)[0]; \
libs_dir=os.path.join(paddle_dir, 'libs'); \
fluid_dir=os.path.join(paddle_dir, 'fluid'); \
out=':'.join([libs_dir, fluid_dir]); print(out); \
    "
    OUTPUT_VARIABLE PADDLE_LIB_DIRS)
message(STATUS PADDLE_LIB_DIRS= ${PADDLE_LIB_DIRS})


###############################################################################
# Add local library
###############################################################################
set(SPEECHX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/speechx)

add_subdirectory(speechx)
