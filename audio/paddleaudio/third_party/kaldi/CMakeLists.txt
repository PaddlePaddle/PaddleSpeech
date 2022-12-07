# checkout the thirdparty/kaldi/base/kaldi-types.h
# compile kaldi without openfst
add_definitions("-DCOMPILE_WITHOUT_OPENFST")

if ((NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/base))
    file(COPY ../../../../speechx/speechx/kaldi/base DESTINATION ${CMAKE_CURRENT_LIST_DIR})
    file(COPY ../../../../speechx/speechx/kaldi/feat DESTINATION ${CMAKE_CURRENT_LIST_DIR})
    file(COPY ../../../../speechx/speechx/kaldi/matrix DESTINATION ${CMAKE_CURRENT_LIST_DIR})
    file(COPY ../../../../speechx/speechx/kaldi/util DESTINATION ${CMAKE_CURRENT_LIST_DIR})
endif()

# kaldi-base
add_library(kaldi-base STATIC
  base/io-funcs.cc
  base/kaldi-error.cc
  base/kaldi-math.cc
  base/kaldi-utils.cc
  base/timer.cc
)
target_include_directories(kaldi-base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# kaldi-matrix
add_library(kaldi-matrix STATIC
  matrix/compressed-matrix.cc
  matrix/matrix-functions.cc
  matrix/kaldi-matrix.cc
  matrix/kaldi-vector.cc
  matrix/optimization.cc
  matrix/packed-matrix.cc
  matrix/qr.cc
  matrix/sparse-matrix.cc
  matrix/sp-matrix.cc
  matrix/srfft.cc
  matrix/tp-matrix.cc
)
target_include_directories(kaldi-matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT MSVC)
    target_link_libraries(kaldi-matrix PUBLIC kaldi-base libopenblas)
else()
    target_link_libraries(kaldi-matrix PUBLIC kaldi-base openblas)
endif()

# kaldi-util
add_library(kaldi-util STATIC
  util/kaldi-holder.cc
  util/kaldi-io.cc
  util/kaldi-semaphore.cc
  util/kaldi-table.cc
  util/kaldi-thread.cc
  util/parse-options.cc
  util/simple-io-funcs.cc
  util/simple-options.cc
  util/text-utils.cc
)
target_include_directories(kaldi-util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kaldi-util PUBLIC kaldi-base kaldi-matrix)

# kaldi-feat-common
add_library(kaldi-feat-common STATIC
  feat/cmvn.cc
  feat/feature-functions.cc
  feat/feature-window.cc
  feat/mel-computations.cc
  feat/pitch-functions.cc
  feat/resample.cc
  feat/signal.cc
  feat/wave-reader.cc
)
target_include_directories(kaldi-feat-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kaldi-feat-common PUBLIC kaldi-base kaldi-matrix kaldi-util)


# kaldi-mfcc
add_library(kaldi-mfcc STATIC
  feat/feature-mfcc.cc
)
target_include_directories(kaldi-mfcc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kaldi-mfcc PUBLIC kaldi-feat-common)


# kaldi-fbank
add_library(kaldi-fbank STATIC
  feat/feature-fbank.cc
)
target_include_directories(kaldi-fbank PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kaldi-fbank PUBLIC kaldi-feat-common)


set(KALDI_LIBRARIES
  ${CMAKE_CURRENT_BINARY_DIR}/libkaldi-base.a
  ${CMAKE_CURRENT_BINARY_DIR}/libkaldi-matrix.a
  ${CMAKE_CURRENT_BINARY_DIR}/libkaldi-util.a
  ${CMAKE_CURRENT_BINARY_DIR}/libkaldi-feat-common.a
  ${CMAKE_CURRENT_BINARY_DIR}/libkaldi-mfcc.a
  ${CMAKE_CURRENT_BINARY_DIR}/libkaldi-fbank.a
)

add_library(libkaldi INTERFACE)
add_dependencies(libkaldi kaldi-base kaldi-matrix kaldi-util kaldi-feat-common kaldi-mfcc kaldi-fbank)
target_include_directories(libkaldi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (APPLE)
    target_link_libraries(libkaldi INTERFACE ${KALDI_LIBRARIES} libopenblas ${GFORTRAN_LIBRARIES_DIR}/libgfortran.a ${GFORTRAN_LIBRARIES_DIR}/libquadmath.a ${GFORTRAN_LIBRARIES_DIR}/libgcc_s.1.1.dylib)
elseif (MSVC)
    target_link_libraries(libkaldi INTERFACE kaldi-base kaldi-matrix kaldi-util kaldi-feat-common kaldi-mfcc kaldi-fbank openblas)
else()
    target_link_libraries(libkaldi INTERFACE -Wl,--start-group -Wl,--whole-archive ${KALDI_LIBRARIES} libopenblas.a gfortran -Wl,--no-whole-archive -Wl,--end-group)
endif()

target_compile_definitions(libkaldi INTERFACE "-DCOMPILE_WITHOUT_OPENFST")
