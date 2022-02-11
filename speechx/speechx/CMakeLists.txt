cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speechx LANGUAGES CXX)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/kaldi
)
add_subdirectory(kaldi)

add_executable(mfcc-test codelab/feat_test/feature-mfcc-test.cc)
target_link_libraries(mfcc-test kaldi-mfcc)
