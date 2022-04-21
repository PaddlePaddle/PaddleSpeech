cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(bin_name linear-spectrogram-wo-db-norm-ol)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} frontend kaldi-util kaldi-feat-common gflags glog)


set(bin_name cmvn-json2kaldi)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} utils kaldi-util kaldi-matrix gflags glog ${DEPS})