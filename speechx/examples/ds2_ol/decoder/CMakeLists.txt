cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(bin_name ctc-prefix-beam-search-decoder-ol)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})


set(bin_name wfst-decoder-ol)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util kaldi-decoder ${DEPS})


set(bin_name nnet-logprob-decoder-test)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})

add_executable(recognizer_test_main ${CMAKE_CURRENT_SOURCE_DIR}/recognizer_test_main.cc)
target_include_directories(recognizer_test_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(recognizer_test_main PUBLIC frontend kaldi-feat-common nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util kaldi-decoder ${DEPS})
