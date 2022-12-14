include(FetchContent)
set(openfst_PREFIX_DIR ${fc_patch}/openfst)
set(openfst_SOURCE_DIR ${fc_patch}/openfst-src)
set(openfst_BINARY_DIR ${fc_patch}/openfst-build)

ExternalProject_Add(openfst
  URL               https://paddleaudio.bj.bcebos.com/build/openfst_1.7.2.zip
  URL_HASH          SHA256=ffc56931025579a8af3515741c0f3b0fc3a854c023421472c07ca0c6389c75e6
  PREFIX            ${openfst_PREFIX_DIR} 
  SOURCE_DIR        ${openfst_SOURCE_DIR}
  BINARY_DIR        ${openfst_BINARY_DIR}
  CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure --prefix=${openfst_PREFIX_DIR}
                      "CPPFLAGS=-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR}"
                      "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                      "LIBS=-lgflags_nothreads -lglog -lpthread"
  COMMAND           ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
  BUILD_COMMAND     make -j 4
)
link_directories(${openfst_PREFIX_DIR}/lib)
include_directories(${openfst_PREFIX_DIR}/include)
