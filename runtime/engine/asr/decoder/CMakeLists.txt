set(srcs)
list(APPEND srcs
  ctc_prefix_beam_search_decoder.cc
  ctc_tlg_decoder.cc
)

add_library(decoder STATIC ${srcs})
target_link_libraries(decoder PUBLIC utils fst frontend nnet kaldi-decoder)

# test
set(TEST_BINS 
  ctc_prefix_beam_search_decoder_main
  ctc_tlg_decoder_main
)

foreach(bin_name IN LISTS TEST_BINS)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  target_link_libraries(${bin_name} nnet decoder fst utils  libgflags_nothreads.so glog kaldi-base kaldi-matrix kaldi-util)
  target_compile_options(${bin_name}  PRIVATE ${PADDLE_COMPILE_FLAGS})
  target_include_directories(${bin_name}  PRIVATE ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
  target_link_libraries(${bin_name}  ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS} -ldl)
endforeach()

