cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


add_executable(mfcc-test ${CMAKE_CURRENT_SOURCE_DIR}/feature-mfcc-test.cc)
target_include_directories(mfcc-test PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(mfcc-test kaldi-mfcc)

add_executable(linear-spectrogram-main ${CMAKE_CURRENT_SOURCE_DIR}/linear-spectrogram-main.cc)
target_include_directories(linear-spectrogram-main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(linear-spectrogram-main frontend kaldi-util kaldi-feat-common gflags glog)