
add_library(kaldi-lat
determinize-lattice-pruned.cc
lattice-functions.cc
)
target_link_libraries(kaldi-lat PUBLIC kaldi-util)