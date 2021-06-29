cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(deepspeech VERSION 0.1)

set(CMAKE_VERBOSE_MAKEFILE on)
# set std-14
set(CMAKE_CXX_STANDARD 14)

# include file 
include(FetchContent)
include(ExternalProject)
# fc_patch dir
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_patch "fc_patch" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_patch})


###############################################################################
# Option Configurations
###############################################################################
# option configurations 
option(TEST_DEBUG "option for debug" OFF)


###############################################################################
# Include third party
###############################################################################
# #example for include third party
# FetchContent_Declare()
# # FetchContent_MakeAvailable was not added until CMake 3.14
# FetchContent_MakeAvailable()
# include_directories()

# ABSEIL-CPP
include(FetchContent)
FetchContent_Declare(
  absl
  GIT_REPOSITORY "https://github.com/abseil/abseil-cpp.git"
  GIT_TAG "20210324.1"
)
FetchContent_MakeAvailable(absl)

# libsndfile
include(FetchContent)
FetchContent_Declare(
  libsndfile
  GIT_REPOSITORY "https://github.com/libsndfile/libsndfile.git"
  GIT_TAG "1.0.31"
)
FetchContent_MakeAvailable(libsndfile)


###############################################################################
# Add local library
###############################################################################
# system lib 
find_package()
# if dir have CmakeLists.txt 
add_subdirectory()
# if dir do not have CmakeLists.txt 
add_library(lib_name STATIC file.cc)
target_link_libraries(lib_name item0 item1)
add_dependencies(lib_name depend-target)


###############################################################################
# Library installation
###############################################################################
install()


###############################################################################
# Build binary file
###############################################################################
add_executable()
target_link_libraries()

