set(openfst_PREFIX_DIR ${fc_patch}/openfst)
set(openfst_SOURCE_DIR ${fc_patch}/openfst-src)
set(openfst_BINARY_DIR ${fc_patch}/openfst-build)

include(FetchContent)
# openfst Acknowledgments:
#Cyril Allauzen, Michael Riley, Johan Schalkwyk, Wojciech Skut and Mehryar Mohri, 
#"OpenFst: A General and Efficient Weighted Finite-State Transducer Library", 
#Proceedings of the Ninth International Conference on Implementation and 
#Application of Automata, (CIAA 2007), volume 4783 of Lecture Notes in 
#Computer Science, pages 11-23. Springer, 2007. http://www.openfst.org.

set(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD 1 # Wrap download in script to log output
    LOG_UPDATE 1 # Wrap update in script to log output
    LOG_CONFIGURE 1# Wrap configure in script to log output
    LOG_BUILD 1 # Wrap build in script to log output
    LOG_TEST 1 # Wrap test in script to log output
    LOG_INSTALL 1 # Wrap install in script to log output
)

ExternalProject_Add(openfst
  URL               https://paddleaudio.bj.bcebos.com/build/openfst_1.7.2.zip
  URL_HASH          SHA256=ffc56931025579a8af3515741c0f3b0fc3a854c023421472c07ca0c6389c75e6
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX            ${openfst_PREFIX_DIR} 
  SOURCE_DIR        ${openfst_SOURCE_DIR}
  BINARY_DIR        ${openfst_BINARY_DIR}
  BUILD_ALWAYS      0
  CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure --prefix=${openfst_PREFIX_DIR}
                      "CPPFLAGS=-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR}"
                      "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                      "LIBS=-lgflags_nothreads -lglog -lpthread"
  COMMAND           ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
  BUILD_COMMAND     make -j 4
)
link_directories(${openfst_PREFIX_DIR}/lib)
include_directories(${openfst_PREFIX_DIR}/include)


message(STATUS "OpenFST inc dir: ${openfst_PREFIX_DIR}/include")
message(STATUS "OpenFST lib dir: ${openfst_PREFIX_DIR}/lib")
