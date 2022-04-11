cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


add_executable(mfcc-test ${CMAKE_CURRENT_SOURCE_DIR}/feature-mfcc-test.cc)
target_include_directories(mfcc-test PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(mfcc-test kaldi-mfcc)

add_executable(linear_spectrogram_main ${CMAKE_CURRENT_SOURCE_DIR}/linear_spectrogram_main.cc)
target_include_directories(linear_spectrogram_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(linear_spectrogram_main frontend kaldi-util kaldi-feat-common gflags glog)

add_executable(linear_spectrogram_without_db_norm_main ${CMAKE_CURRENT_SOURCE_DIR}/linear_spectrogram_without_db_norm_main.cc)
target_include_directories(linear_spectrogram_without_db_norm_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(linear_spectrogram_without_db_norm_main frontend kaldi-util kaldi-feat-common gflags glog)

add_executable(cmvn_json2binary_main ${CMAKE_CURRENT_SOURCE_DIR}/cmvn_json2binary_main.cc)
target_include_directories(cmvn_json2binary_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(cmvn_json2binary_main utils kaldi-util kaldi-matrix gflags glog)
