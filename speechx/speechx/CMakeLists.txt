cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speechx LANGUAGES CXX)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/kaldi
)
add_subdirectory(kaldi)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/utils
)
add_subdirectory(utils)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/frontend
)
add_subdirectory(frontend)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/nnet
)
add_subdirectory(nnet)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/decoder
)
add_subdirectory(decoder)

link_libraries(dl)

add_executable(mfcc-test codelab/feat_test/feature-mfcc-test.cc)
target_link_libraries(mfcc-test kaldi-mfcc ${MATH_LIB})

add_executable(linear_spectrogram_main codelab/feat_test/linear_spectrogram_main.cc)
target_link_libraries(linear_spectrogram_main frontend kaldi-util kaldi-feat-common gflags glog)

add_executable(offline_decoder_main codelab/decoder_test/offline_decoder_main.cc)
target_link_libraries(offline_decoder_main PUBLIC nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})

add_executable(model_test codelab/nnet_test/model_test.cc)
target_link_libraries(model_test PUBLIC nnet gflags ${DEPS})
