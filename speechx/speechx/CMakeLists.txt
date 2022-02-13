cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speechx LANGUAGES CXX)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/kaldi
)
add_subdirectory(kaldi)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/frontend
)
add_subdirectory(frontend)

add_executable(mfcc-test codelab/feat_test/feature-mfcc-test.cc)
target_link_libraries(mfcc-test kaldi-mfcc)

add_executable(linear_spectrogram_main codelab/feat_test/linear_spectrogram_main.cc)
target_link_libraries(linear_spectrogram_main frontend kaildi-util kaldi-feat)
