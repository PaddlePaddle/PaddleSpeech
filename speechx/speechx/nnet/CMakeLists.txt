project(nnet)

add_library(nnet STATIC
  decodable.cc
  ds2_nnet.cc
)
target_link_libraries(nnet absl::strings)

set(bin_name ds2_nnet_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} utils kaldi-util kaldi-matrix gflags glog nnet ${DEPS})