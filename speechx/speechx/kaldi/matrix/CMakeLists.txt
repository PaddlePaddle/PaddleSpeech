
add_library(kaldi-matrix
compressed-matrix.cc
kaldi-matrix.cc
kaldi-vector.cc
matrix-functions.cc
optimization.cc
packed-matrix.cc
qr.cc
sparse-matrix.cc
sp-matrix.cc
srfft.cc
tp-matrix.cc
)

target_link_libraries(kaldi-matrix gfortran kaldi-base libopenblas.a)
