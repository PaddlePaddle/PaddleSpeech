
add_executable(arpa2fst ${CMAKE_CURRENT_SOURCE_DIR}/arpa2fst.cc)
target_include_directories(arpa2fst PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(arpa2fst PUBLIC kaldi-lm glog gflags fst)
