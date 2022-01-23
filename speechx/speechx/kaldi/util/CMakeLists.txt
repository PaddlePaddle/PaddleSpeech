add_library(kaldi-util
  kaldi-holder.cc
  kaldi-io.cc
  kaldi-semaphore.cc
  kaldi-table.cc
  kaldi-thread.cc
  parse-options.cc
  simple-io-funcs.cc
  simple-options.cc
  text-utils.cc
)
target_link_libraries(kaldi-util PUBLIC kaldi-base kaldi-matrix)