# >=3.17 support -DCMAKE_FIND_DEBUG_MODE=ON
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableCMP0048.cmake")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(system)

project(paddlespeech VERSION 0.1)

set(PPS_VERSION_MAJOR 1)
set(PPS_VERSION_MINOR 0)
set(PPS_VERSION_PATCH 0)
set(PPS_VERSION "${PPS_VERSION_MAJOR}.${PPS_VERSION_MINOR}.${PPS_VERSION_PATCH}")

# fc_patch dir
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_patch "fc_patch" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_patch})

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_FIND_DEBUG_MODE OFF)
set(PPS_CXX_STANDARD 14)

# set std-14
set(CMAKE_CXX_STANDARD ${PPS_CXX_STANDARD})

# Ninja Generator will set CMAKE_BUILD_TYPE to Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# find_* e.g. find_library work when Cross-Compiling
if(ANDROID)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

# install dir into `build/install`
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)

include(FetchContent)
include(ExternalProject)
###############################################################################
# Option Configurations
###############################################################################
# https://github.com/google/brotli/pull/655
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

option(WITH_ASR "build asr" ON)
option(WITH_CLS "build cls" ON)
option(WITH_VAD "build vad" ON)

option(WITH_GPU "NNet using GPU." OFF)

option(WITH_PROFILING "enable c++ profling" OFF)
option(WITH_TESTING "unit test" ON)

###############################################################################
# Include Third Party
###############################################################################
include(gflags)

include(glog)

# gtest
if(WITH_TESTING)
    include(gtest) # download, build, install gtest
endif()

# fastdeploy
if(NOT WITH_ASR)
    include(fastdeploy)
endif()

if(WITH_ASR)
    # openfst
    include(openfst)
    add_dependencies(openfst gflags extern_glog)
endif()

###############################################################################
# Find Package
###############################################################################
# https://github.com/Kitware/CMake/blob/v3.1.0/Modules/FindThreads.cmake#L207
find_package(Threads REQUIRED)

if(WITH_ASR)
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
    # -L/workspace/DeepSpeech-2.x/engine/venv/lib/python3.7/site-packages/paddle/libs -L/workspace/DeepSpeech-2.x/speechx/venv/lib/python3.7/site-packages/paddle/fluid -l:libpaddle.so -l:libdnnl.so.2 -l:libiomp5.so
    set(EXECUTE_COMMAND "import os"
        "import paddle"
        "include_dir = paddle.sysconfig.get_include()"
        "paddle_dir=os.path.split(include_dir)[0]"
        "libs_dir=os.path.join(paddle_dir, 'libs')"
        "fluid_dir=os.path.join(paddle_dir, 'fluid')"
        "out=' '.join([\"-L\" + libs_dir, \"-L\" + fluid_dir])"
        "out += \" -l:libpaddle.so -l:libdnnl.so.2 -l:libiomp5.so\"; print(out)"
    )
    execute_process(
        COMMAND python -c "${EXECUTE_COMMAND}"
        OUTPUT_VARIABLE PADDLE_LINK_FLAGS
        RESULT_VARIABLE SUCESS)

    message(STATUS PADDLE_LINK_FLAGS= ${PADDLE_LINK_FLAGS})
    string(STRIP ${PADDLE_LINK_FLAGS} PADDLE_LINK_FLAGS)

    # paddle compile option
    # -I/workspace/DeepSpeech-2.x/engine/venv/lib/python3.7/site-packages/paddle/include
    set(EXECUTE_COMMAND "import paddle"
        "include_dir = paddle.sysconfig.get_include()"
        "print(f\"-I{include_dir}\")"
    )
    execute_process(
        COMMAND python -c "${EXECUTE_COMMAND}"
        OUTPUT_VARIABLE PADDLE_COMPILE_FLAGS)
    message(STATUS PADDLE_COMPILE_FLAGS= ${PADDLE_COMPILE_FLAGS})
    string(STRIP ${PADDLE_COMPILE_FLAGS} PADDLE_COMPILE_FLAGS)

    # for LD_LIBRARY_PATH
    # set(PADDLE_LIB_DIRS /workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/libs/)
    set(EXECUTE_COMMAND "import os"
        "import paddle"
        "include_dir=paddle.sysconfig.get_include()"
        "paddle_dir=os.path.split(include_dir)[0]"
        "libs_dir=os.path.join(paddle_dir, 'libs')"
        "fluid_dir=os.path.join(paddle_dir, 'fluid')"
        "out=':'.join([libs_dir, fluid_dir]); print(out)"
        )
    execute_process(
        COMMAND python -c "${EXECUTE_COMMAND}"
        OUTPUT_VARIABLE PADDLE_LIB_DIRS)
    message(STATUS PADDLE_LIB_DIRS= ${PADDLE_LIB_DIRS})
endif()

include(summary)

###############################################################################
# Add local library
###############################################################################
set(ENGINE_ROOT ${CMAKE_SOURCE_DIR}/engine)

add_subdirectory(engine)


###############################################################################
# CPack library
###############################################################################
# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME "paddlespeech_library")
set(CPACK_PACKAGE_VENDOR "paddlespeech")
set(CPACK_PACKAGE_VERSION_MAJOR 1)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_PACKAGE_DESCRIPTION "paddlespeech library")
set(CPACK_PACKAGE_CONTACT "paddlespeech@baidu.com")
set(CPACK_SOURCE_GENERATOR "TGZ")
include (CPack)