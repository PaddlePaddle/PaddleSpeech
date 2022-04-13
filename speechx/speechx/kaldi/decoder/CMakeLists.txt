
add_library(kaldi-decoder
lattice-faster-decoder.cc
lattice-faster-online-decoder.cc
)
target_link_libraries(kaldi-decoder PUBLIC kaldi-lat)
