
add_library(kaldi-lm
  arpa-file-parser.cc
  arpa-lm-compiler.cc
)
target_link_libraries(kaldi-lm PUBLIC kaldi-util)