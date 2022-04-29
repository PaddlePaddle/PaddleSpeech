project(frontend)

add_library(frontend STATIC
  cmvn.cc
  db_norm.cc
  linear_spectrogram.cc
  audio_cache.cc
  feature_cache.cc
  feature_pipeline.cc
  fbank.cc
)

target_link_libraries(frontend PUBLIC kaldi-matrix kaldi-feat-common)
