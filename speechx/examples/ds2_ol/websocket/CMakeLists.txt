cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

add_executable(websocket_server_main ${CMAKE_CURRENT_SOURCE_DIR}/websocket_server_main.cc)
target_include_directories(websocket_server_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(websocket_server_main PUBLIC frontend kaldi-feat-common nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util kaldi-decoder websocket ${DEPS})

add_executable(websocket_client_main ${CMAKE_CURRENT_SOURCE_DIR}/websocket_client_main.cc)
target_include_directories(websocket_client_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(websocket_client_main PUBLIC frontend kaldi-feat-common nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util kaldi-decoder websocket ${DEPS})

