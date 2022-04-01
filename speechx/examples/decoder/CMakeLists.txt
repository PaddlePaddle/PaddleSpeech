cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

add_executable(offline_decoder_sliding_chunk_main ${CMAKE_CURRENT_SOURCE_DIR}/offline_decoder_sliding_chunk_main.cc)
target_include_directories(offline_decoder_sliding_chunk_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(offline_decoder_sliding_chunk_main PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})

add_executable(offline_decoder_main ${CMAKE_CURRENT_SOURCE_DIR}/offline_decoder_main.cc)
target_include_directories(offline_decoder_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(offline_decoder_main PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})

add_executable(decoder_test_main ${CMAKE_CURRENT_SOURCE_DIR}/decoder_test_main.cc)
target_include_directories(decoder_test_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(decoder_test_main PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})

