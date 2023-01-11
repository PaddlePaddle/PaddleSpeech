
add_library(kaldi-matrix
kaldi-matrix.cc
kaldi-vector.cc
)

target_link_libraries(kaldi-matrix kaldi-base)
