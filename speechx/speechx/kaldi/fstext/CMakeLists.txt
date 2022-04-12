
add_library(kaldi-fstext
kaldi-fst-io.cc
)
target_link_libraries(kaldi-fstext PUBLIC kaldi-util)
