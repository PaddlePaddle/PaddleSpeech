cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

add_executable(arpa2fst ${CMAKE_CURRENT_SOURCE_DIR}/arpa2fst.cc)
target_include_directories(arpa2fst PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(arpa2fst )
