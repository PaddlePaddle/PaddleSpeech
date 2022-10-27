cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(paddlespeech VERSION 0.1)

set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableCMP0048.cmake")

set(CMAKE_VERBOSE_MAKEFILE on)

# set std-14
set(CMAKE_CXX_STANDARD 14)

# cmake dir
set(speechx_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

# Modules
list(APPEND CMAKE_MODULE_PATH ${speechx_cmake_dir})
include(FetchContent)
include(ExternalProject)

# fc_patch dir
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_patch "fc_patch" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_patch})

# compiler option
# Keep the same with openfst, -fPIC or -fpic
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14 -pthread -fPIC -O0 -Wall -g")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} --std=c++14 -pthread -fPIC -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} --std=c++14 -pthread -fPIC -O3 -Wall")

###############################################################################
# Option Configurations
###############################################################################
option(TEST_DEBUG "option for debug" OFF)
option(USE_PROFILING "enable c++ profling" OFF)

option(USING_U2  "compile u2 model." ON)
option(USING_DS2 "compile with ds2 model." ON)

option(USING_GPU "u2 compute on GPU." OFF)

###############################################################################
# Include third party
###############################################################################
# example for include third party
# FetchContent_MakeAvailable was not added until CMake 3.14
# FetchContent_MakeAvailable()
# include_directories()

# gflags
include(gflags)

# glog
include(glog)

# gtest
include(gtest)

# ABSEIL-CPP
include(absl)

# libsndfile
include(libsndfile)

# boost
# include(boost) # not work
set(boost_SOURCE_DIR ${fc_patch}/boost-src)
set(BOOST_ROOT ${boost_SOURCE_DIR})
include_directories(${boost_SOURCE_DIR})
link_directories(${boost_SOURCE_DIR}/stage/lib)

# Eigen
include(eigen)
find_package(Eigen3 REQUIRED)

# Kenlm
include(kenlm)
add_dependencies(kenlm eigen boost)

#openblas
include(openblas)

# openfst
include(openfst)
add_dependencies(openfst gflags glog)


# paddle lib
include(paddleinference)


# paddle core.so
find_package(Threads REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Python3 REQUIRED)
find_package(pybind11 CONFIG)

message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "Pybind11_INCLUDES = ${pybind11_INCLUDE_DIRS}, pybind11_LIBRARIES=${pybind11_LIBRARIES}, pybind11_DEFINITIONS=${pybind11_DEFINITIONS}")

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
    COMMAND python -c " \
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
# system lib 
#find_package()
# if dir have CmakeLists.txt 
#add_subdirectory(speechx)
# if dir do not have CmakeLists.txt 
#add_library(lib_name STATIC file.cc)
#target_link_libraries(lib_name item0 item1)
#add_dependencies(lib_name depend-target)

set(SPEECHX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/speechx)

add_subdirectory(speechx)
