# checkout the thirdparty/kaldi/base/kaldi-types.h
# compile kaldi without openfst
add_definitions("-DCOMPILE_WITHOUT_OPENFST")

# function (define_library name source include_dirs link_libraries compile_defs)
#   add_library(${name} INTERFACE ${source})
#   target_include_directories(${name} INTERFACE ${include_dirs})
#   target_link_libraries(${name} INTERFACE ${link_libraries})
#   target_compile_definitions(${name} INTERFACE ${compile_defs})
#   set_target_properties(${name} PROPERTIES PREFIX "")
#   if (MSVC)
#     set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
#   endif(MSVC)
#   install(
#     TARGETS ${name}
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION lib  # For Windows
#     )
# endfunction()

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
target_link_libraries(kaldi-matrix PUBLIC gfortran kaldi-base libopenblas)


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
target_link_libraries(libkaldi INTERFACE   
  # --whole-archive  for undefined symbol when link static lib into shared lib
  -Wl,--start-group -Wl,--whole-archive 
  ${KALDI_LIBRARIES}
  libopenblas
  gfortran
  -Wl,--no-whole-archive -Wl,--end-group
)
target_compile_definitions(libkaldi INTERFACE "-DCOMPILE_WITHOUT_OPENFST")
