# checkout the thirdparty/kaldi/base/kaldi-types.h
# compile kaldi without openfst
add_definitions("-DCOMPILE_WITHOUT_OPENFST")

# include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/kaldi)
# include_directories(/usr/include/python3.7m)

set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

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
add_library(kaldi-base INTERFACE
  base/io-funcs.cc
  base/kaldi-error.cc
  base/kaldi-math.cc
  base/kaldi-utils.cc
  base/timer.cc
)
target_include_directories(kaldi-base INTERFACE ${INSTALL_DIR}/base)
target_compile_definitions(kaldi-base INTERFACE "-DCOMPILE_WITHOUT_OPENFST")

# kaldi-matrix
add_library(kaldi-matrix INTERFACE
  matrix/compressed-matrix.cc
  matrix/kaldi-matrix.cc
  matrix/kaldi-vector.cc
  matrix/matrix-functions.cc
  matrix/optimization.cc
  matrix/packed-matrix.cc
  matrix/qr.cc
  matrix/sparse-matrix.cc
  matrix/sp-matrix.cc
  matrix/srfft.cc
  matrix/tp-matrix.cc
)
target_include_directories(kaldi-matrix INTERFACE ${INSTALL_DIR}/matrix)
target_link_libraries(kaldi-matrix INTERFACE gfortran kaldi-base libopenblas.a)

# kaldi-util
add_library(kaldi-util INTERFACE
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
target_include_directories(kaldi-util INTERFACE ${INSTALL_DIR}/util)
target_link_libraries(kaldi-util INTERFACE kaldi-base kaldi-matrix)

# kaldi-feat-common
add_library(kaldi-feat-common INTERFACE
  feat/wave-reader.cc
  feat/signal.cc
  feat/feature-functions.cc
  feat/feature-window.cc
  feat/resample.cc
  feat/mel-computations.cc
  feat/cmvn.cc
)
target_include_directories(kaldi-feat-common INTERFACE ${INSTALL_DIR}/feat)
target_link_libraries(kaldi-feat-common INTERFACE kaldi-base kaldi-matrix kaldi-util)

# kaldi-mfcc
add_library(kaldi-mfcc INTERFACE
  feat/feature-mfcc.cc
)
target_include_directories(kaldi-mfcc INTERFACE ${INSTALL_DIR}/feat)
target_link_libraries(kaldi-mfcc INTERFACE kaldi-feat-common)

# kaldi-fbank
add_library(kaldi-fbank INTERFACE
  feat/feature-fbank.cc
)
target_include_directories(kaldi-fbank INTERFACE ${INSTALL_DIR}/feat)
target_link_libraries(kaldi-fbank INTERFACE kaldi-feat-common)