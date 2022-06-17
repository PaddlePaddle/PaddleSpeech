project(kaldi)


add_library(kaldi-base  
  base/io-funcs.cc
  base/kaldi-error.cc
  base/kaldi-math.cc
  base/kaldi-utils.cc
  base/timer.cc
)

add_library(kaldi-util
  util/kaldi-holder.cc
  util/kaldi-io.cc
  util/kaldi-semaphore.cc
  util/kaldi-table.cc
  util/kaldi-thread.cc
  util/parse-options.cc
  util/simple-io-funcs.cc
  util/simple-options.cc
  util/text-utils.cc
)
target_link_libraries(kaldi-util PUBLIC kaldi-base kaldi-matrix)

add_library(kaldi-mfcc
  feat/feature-mfcc.cc
)
target_link_libraries(kaldi-mfcc PUBLIC kaldi-feat-common)

add_library(kaldi-fbank
  feat/feature-fbank.cc
)
target_link_libraries(kaldi-fbank PUBLIC kaldi-feat-common)

add_library(kaldi-feat-common
  feat/wave-reader.cc
  feat/signal.cc
  feat/feature-functions.cc
  feat/feature-window.cc
  feat/resample.cc
  feat/mel-computations.cc
  feat/cmvn.cc
)
target_link_libraries(kaldi-feat-common PUBLIC kaldi-base kaldi-matrix kaldi-util)

add_library(kaldi-matrix
  matrix/compressed-matrix.cc
  matrix/kaldi-matrix.cc
  matrix/kaldi-vector.cc
  matrix/matrix-functions.cc
  matrix/optimization.cc
  matrix/packed-matrix.cc
  matrix/qr.cc
  matrix/sparse-matrix.cc
  matrix/sp-matrix.cc
  matrix/srfft.cc
  matrix/tp-matrix.cc
)
target_link_libraries(kaldi-matrix gfortran kaldi-base libopenblas.a)



#add_subdirectory(base)
#add_subdirectory(util)
#add_subdirectory(feat)
#add_subdirectory(matrix)
