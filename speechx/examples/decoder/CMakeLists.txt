cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

add_executable(offline-decoder-main ${CMAKE_CURRENT_SOURCE_DIR}/offline-decoder-main.cc)
target_include_directories(offline-decoder-main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(offline-decoder-main PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})