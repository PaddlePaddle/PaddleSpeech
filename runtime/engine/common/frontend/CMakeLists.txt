add_library(kaldi-native-fbank-core 
  feature-fbank.cc
  feature-functions.cc
  feature-window.cc
  fftsg.c
  mel-computations.cc
  rfft.cc
)
target_link_libraries(kaldi-native-fbank-core PUBLIC utils base)

add_library(frontend STATIC
  cmvn.cc
  audio_cache.cc
  feature_cache.cc
  feature_pipeline.cc
  assembler.cc
  wave-reader.cc
)
target_link_libraries(frontend PUBLIC kaldi-native-fbank-core utils base)

set(BINS 
  compute_fbank_main
)

foreach(bin_name IN LISTS BINS)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  # https://github.com/Kitware/CMake/blob/v3.1.0/Modules/FindThreads.cmake#L207
  target_link_libraries(${bin_name} PUBLIC frontend base utils kaldi-util gflags Threads::Threads extern_glog)
endforeach()