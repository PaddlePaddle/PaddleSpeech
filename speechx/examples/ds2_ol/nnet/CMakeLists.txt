cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(bin_name ds2-model-ol-test)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} PUBLIC nnet gflags glog ${DEPS})