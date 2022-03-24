project(decoder)

include_directories(${CMAKE_CURRENT_SOURCE_DIR/ctc_decoders})
add_library(decoder STATIC
  ctc_beam_search_decoder.cc
  ctc_decoders/decoder_utils.cpp
  ctc_decoders/path_trie.cpp
  ctc_decoders/scorer.cpp
  ctc_tlg_decoder.cc
)
target_link_libraries(decoder PUBLIC kenlm utils fst)
