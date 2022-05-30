project(frontend)

add_library(frontend STATIC
  cmvn.cc
  db_norm.cc
  linear_spectrogram.cc
  audio_cache.cc
  feature_cache.cc
  feature_pipeline.cc
  fbank.cc
  dispenser.cc
)
target_link_libraries(frontend PUBLIC kaldi-matrix kaldi-feat-common kaldi-fbank)



set(bin_name cmvn_json2kaldi_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} utils kaldi-util kaldi-matrix gflags glog)

set(BINS 
  compute_linear_spectrogram_main
  compute_fbank_main
)

foreach(bin_name IN LISTS BINS)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  target_link_libraries(${bin_name} PUBLIC frontend utils kaldi-util gflags glog)
endforeach()
