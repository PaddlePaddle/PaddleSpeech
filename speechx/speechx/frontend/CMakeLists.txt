project(frontend)

add_library(frontend STATIC
  normalizer.cc
  linear_spectrogram.cc  
)

target_link_libraries(frontend PUBLIC kaldi-matrix)