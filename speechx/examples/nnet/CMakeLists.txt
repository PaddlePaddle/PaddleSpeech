cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

add_executable(pp-model-test ${CMAKE_CURRENT_SOURCE_DIR}/pp-model-test.cc)
target_include_directories(pp-model-test PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(pp-model-test PUBLIC nnet gflags ${DEPS})