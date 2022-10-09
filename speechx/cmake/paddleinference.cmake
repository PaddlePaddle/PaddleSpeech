set(paddle_SOURCE_DIR ${fc_patch}/paddle-lib)
set(paddle_PREFIX_DIR ${fc_patch}/paddle-lib-prefix)
# ExternalProject_Add(paddle
#   URL      https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz
#   URL_HASH SHA256=7c6399e778c6554a929b5a39ba2175e702e115145e8fa690d2af974101d98873
#   PREFIX            ${paddle_PREFIX_DIR} 
#   SOURCE_DIR        ${paddle_SOURCE_DIR}
#   CONFIGURE_COMMAND ""
#   BUILD_COMMAND     ""
#   INSTALL_COMMAND   ""
# )

include(FetchContent)
FetchContent_Declare(
  paddle
  URL      https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz
  URL_HASH SHA256=7c6399e778c6554a929b5a39ba2175e702e115145e8fa690d2af974101d98873
  PREFIX            ${paddle_PREFIX_DIR} 
  SOURCE_DIR        ${paddle_SOURCE_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)
FetchContent_MakeAvailable(paddle)

set(PADDLE_LIB_THIRD_PARTY_PATH "${paddle_SOURCE_DIR}/third_party/install/")

include_directories("${paddle_SOURCE_DIR}/paddle/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/include")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/lib")
link_directories("${paddle_SOURCE_DIR}/paddle/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}mklml/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn/lib")

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

# global vars
set(DEPS ${paddle_SOURCE_DIR}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX} CACHE INTERNAL "deps")
set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags protobuf xxhash cryptopp
      ${EXTERNAL_LIB} CACHE INTERNAL "deps")
message(STATUS "Deps libraries: ${DEPS}")
