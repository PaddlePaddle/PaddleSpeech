add_library(kaldi-mfcc
  feature-mfcc.cc
)
target_link_libraries(kaldi-mfcc PUBLIC kaldi-feat-common)

add_library(fbank
  feature-fbank.cc
)
target_link_libraries(fbank PUBLIC kaldi-feat-common)

add_library(kaldi-feat-common
  wave-reader.cc
  signal.cc
  feature-functions.cc
  feature-window.cc
  resample.cc
  mel-computations.cc
  cmvn.cc
)
target_link_libraries(kaldi-feat-common PUBLIC kaldi-base kaldi-matrix kaldi-util)
