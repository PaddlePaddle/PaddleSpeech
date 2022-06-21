
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(kaldi_feature
  kaldi_feature.cc
  kaldi_feature_wrapper.cc
)
target_link_libraries(kaldi_feature kaldi-fbank)

pybind11_add_module(kaldi_featurepy kaldi_feature.cc kaldi_feature_wrapper.cc)
target_link_libraries(kaldi_featurepy PRIVATE kaldi_feature)

set(bin_name kaldi_feature_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${bin_name} PUBLIC kaldi_feature python3.7m)
