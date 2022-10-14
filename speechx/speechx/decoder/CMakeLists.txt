project(decoder)

include_directories(${CMAKE_CURRENT_SOURCE_DIR/ctc_decoders})
add_library(decoder STATIC
  ctc_decoders/decoder_utils.cpp
  ctc_decoders/path_trie.cpp
  ctc_decoders/scorer.cpp
  ctc_beam_search_decoder.cc
  ctc_prefix_beam_search_decoder.cc
  ctc_tlg_decoder.cc
  recognizer.cc
  u2_recognizer.cc
)
target_link_libraries(decoder PUBLIC kenlm utils fst frontend nnet kaldi-decoder absl::strings)

# test
set(BINS 
  ctc_beam_search_decoder_main
  nnet_logprob_decoder_main
  recognizer_main
  ctc_tlg_decoder_main
)

foreach(bin_name IN LISTS BINS)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  target_link_libraries(${bin_name} PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})
endforeach()


# u2
set(TEST_BINS 
  u2_recognizer_main
  ctc_prefix_beam_search_decoder_main
)

foreach(bin_name IN LISTS TEST_BINS)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  target_link_libraries(${bin_name} nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util)
  target_compile_options(${bin_name}  PRIVATE ${PADDLE_COMPILE_FLAGS})
  target_include_directories(${bin_name}  PRIVATE ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
  target_link_libraries(${bin_name}  ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})
endforeach()