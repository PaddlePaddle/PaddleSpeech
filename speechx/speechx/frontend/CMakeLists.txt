project(frontend)

add_library(frontend STATIC
  normalizer.cc
  linear_spectrogram.cc
  raw_audio.cc
  feature_cache.cc
)

target_link_libraries(frontend PUBLIC kaldi-matrix)
