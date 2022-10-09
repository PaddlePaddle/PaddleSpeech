project(decoder)

include_directories(${CMAKE_CURRENT_SOURCE_DIR/ctc_decoders})
add_library(decoder STATIC
  ctc_beam_search_decoder.cc
  ctc_decoders/decoder_utils.cpp
  ctc_decoders/path_trie.cpp
  ctc_decoders/scorer.cpp
  ctc_tlg_decoder.cc
  recognizer.cc
)
target_link_libraries(decoder PUBLIC kenlm utils fst frontend nnet kaldi-decoder)

set(BINS 
  ctc_prefix_beam_search_decoder_main
  nnet_logprob_decoder_main
  recognizer_main
  tlg_decoder_main
)

message(STATUS "xxxxxxxxxx: " ${DEPS})
foreach(bin_name IN LISTS BINS)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  target_link_libraries(${bin_name} PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})
endforeach()
