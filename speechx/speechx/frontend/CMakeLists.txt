project(frontend)

add_library(frontend
  normalizer.cc
  linear_spectrogram.cc  
)

target_link_libraries(frontend kaldi-matrix)