cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(paddlespeech VERSION 0.1)

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
include_directories(${absl_SOURCE_DIR})

# libsndfile
#include(FetchContent)
#FetchContent_Declare(
#  libsndfile
#  GIT_REPOSITORY "https://github.com/libsndfile/libsndfile.git"
#  GIT_TAG "1.0.31"
#)
#FetchContent_MakeAvailable(libsndfile)

set(boost_SOURCE_DIR ${fc_patch}/boost-src)
set(BOOST_ROOT ${boost_SOURCE_DIR})
include(FetchContent)
FetchContent_Declare(
  kenlm
  GIT_REPOSITORY "https://github.com/kpu/kenlm.git"
  GIT_TAG "df2d717e95183f79a90b2fa6e4307083a351ca6a"
)
FetchContent_MakeAvailable(kenlm)
include_directories(${kenlm_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC -O0 -Wall -g")
# gflags
FetchContent_Declare(
  gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.1.zip
  URL_HASH SHA256=4e44b69e709c826734dbbbd5208f61888a2faf63f239d73d8ba0011b2dccc97a
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)

# glog
FetchContent_Declare(
  glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_BINARY_DIR} ${glog_SOURCE_DIR}/src)

# gtest
#FetchContent_Declare(googletest
  #URL      https://github.com/google/googletest/archive/release-1.10.0.zip
  #URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
#)
#FetchContent_MakeAvailable(googletest)

# openfst
set(openfst_SOURCE_DIR ${fc_patch}/openfst-src)
set(openfst_BINARY_DIR ${fc_patch}/openfst-build)
set(openfst_PREFIX_DIR ${fc_patch}/openfst-subbuild/openfst-populate-prefix)
ExternalProject_Add(openfst
  URL               https://github.com/mjansche/openfst/archive/refs/tags/1.7.2.zip
  URL_HASH          SHA256=ffc56931025579a8af3515741c0f3b0fc3a854c023421472c07ca0c6389c75e6
  #PREFIX            ${openfst_PREFIX_DIR} 
  SOURCE_DIR        ${openfst_SOURCE_DIR}
  BINARY_DIR        ${openfst_BINARY_DIR}
  CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure --prefix=${openfst_PREFIX_DIR}
                      "CPPFLAGS=-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR}"
                      "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                      "LIBS=-lgflags_nothreads -lglog -lpthread"
  COMMAND           ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
  BUILD_COMMAND     make -j 4
)
add_dependencies(openfst gflags glog)
link_directories(${openfst_PREFIX_DIR}/lib)
include_directories(${openfst_PREFIX_DIR}/include)

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

#openblas
set(OpenBLAS_INSTALL_PREFIX ${fc_patch}/openblas-install)
#set(OpenBLAS_INSTALL_PREFIX ${fc_patch}/OpenBLAS)
#set(OpenBLAS_SOURCE_DIR ${fc_patch}/OpenBLAS-src)
#set(OpenBLAS_PREFIX ${fc_patch}/OpenBLAS-prefix)
#ExternalProject_Add(
  #OpenBLAS
  #GIT_REPOSITORY https://github.com/xianyi/OpenBLAS
  #GIT_TAG        1ef97c470ccf23d06d3bb0597c52df4a2e9093b9
  #GIT_SHALLOW TRUE
  #GIT_PROGRESS TRUE
  #PREFIX      ${OpenBLAS_PREFIX}
  #SOURCE_DIR  ${OpenBLAS_SOURCE_DIR}
  #BUILD_IN_SOURCE TRUE
  #CONFIGURE_COMMAND ""
  #BUILD_COMMAND make
  #INSTALL_COMMAND make PREFIX=${OpenBLAS_INSTALL_PREFIX} install
  #UPDATE_DISCONNECTED TRUE
#)
link_directories(${OpenBLAS_INSTALL_PREFIX}/lib)
include_directories(${OpenBLAS_INSTALL_PREFIX}/include)

# gfortan dir in the docker.
link_directories(/usr/local/gcc-8.2/lib64)

add_subdirectory(speechx)

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
