cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(paddlespeech VERSION 0.1)

set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableCMP0048.cmake")

set(CMAKE_VERBOSE_MAKEFILE on)

# set std-14
set(CMAKE_CXX_STANDARD 14)

# cmake dir
set(speechx_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

# Modules
list(APPEND CMAKE_MODULE_PATH ${speechx_cmake_dir}/external)
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
# #find_package(boost REQUIRED PATHS ${BOOST_ROOT})

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
set(paddle_SOURCE_DIR ${fc_patch}/paddle-lib)
set(paddle_PREFIX_DIR ${fc_patch}/paddle-lib-prefix)
ExternalProject_Add(paddle
  URL      https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz
  URL_HASH SHA256=7c6399e778c6554a929b5a39ba2175e702e115145e8fa690d2af974101d98873
  PREFIX            ${paddle_PREFIX_DIR} 
  SOURCE_DIR        ${paddle_SOURCE_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

set(PADDLE_LIB ${fc_patch}/paddle-lib)
include_directories("${PADDLE_LIB}/paddle/include")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/include")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/lib")
link_directories("${PADDLE_LIB}/paddle/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}mklml/lib")

##paddle with mkl
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
include_directories("${MATH_LIB_PATH}/include")
set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
include_directories("${MKLDNN_PATH}/include")
set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
set(EXTERNAL_LIB "-lrt -ldl -lpthread")

set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags protobuf xxhash cryptopp
      ${EXTERNAL_LIB})



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
add_subdirectory(examples)