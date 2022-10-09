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
# option configurations 
option(TEST_DEBUG "option for debug" OFF)


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
