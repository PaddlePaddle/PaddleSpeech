
include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(kaldi_feature
  kaldi_feature.cc
  kaldi_feature_wrapper.cc
)
target_link_libraries(kaldi_feature kaldi-fbank)

pybind11_add_module(kaldi_frontend kaldi_feature.cc kaldi_feature_wrapper.cc)
target_link_libraries(kaldi_frontend PRIVATE kaldi_feature)
