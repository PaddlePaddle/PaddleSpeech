
set(sources
  feature-fbank.cc
  feature-functions.cc
  feature-window.cc
  fftsg.c
  mel-computations.cc
  online-feature.cc
  rfft.cc
)

add_library(kaldi-native-fbank-core ${sources})