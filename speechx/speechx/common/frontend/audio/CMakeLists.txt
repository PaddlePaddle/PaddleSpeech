add_library(kaldi-native-fbank-core 
  feature-fbank.cc
  feature-functions.cc
  feature-window.cc
  fftsg.c
  mel-computations.cc
  rfft.cc
)

add_library(frontend STATIC
  cmvn.cc
  audio_cache.cc
  feature_cache.cc
  feature_pipeline.cc
  assembler.cc
)
target_link_libraries(frontend PUBLIC kaldi-native-fbank-core utils)

set(BINS 
  compute_fbank_main
)

foreach(bin_name IN LISTS BINS)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  target_link_libraries(${bin_name} PUBLIC frontend utils kaldi-util gflags glog kaldi-feat-common)
endforeach()
