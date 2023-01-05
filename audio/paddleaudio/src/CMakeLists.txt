if (MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(APPLE)
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

################################################################################
# libpaddleaudio
################################################################################
set(
  LIBPADDLEAUDIO_SOURCES
  utils.cpp
  )

set(
  LIBPADDLEAUDIO_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  )

set(
  LIBPADDLEAUDIO_LINK_LIBRARIES
  )

set(
  LIBPADDLEAUDIO_COMPILE_DEFINITIONS)

#------------------------------------------------------------------------------#
# START OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#

if(BUILD_SOX)
  list(
    APPEND
    LIBPADDLEAUDIO_LINK_LIBRARIES
    libsox
    )
  list(
    APPEND
    LIBPADDLEAUDIO_SOURCES
    )
  list(
    APPEND
    LIBPADDLEAUDIO_COMPILE_DEFINITIONS
    INCLUDE_SOX
    )
endif()


if(BUILD_KALDI)
  list(
    APPEND
    LIBPADDLEAUDIO_LINK_LIBRARIES
    kaldi-native-fbank-core
  )
  list(
    APPEND
    LIBPADDLEAUDIO_COMPILE_DEFINITIONS
    INCLUDE_KALDI
    COMPILE_WITHOUT_OPENFST
  )
endif()

#------------------------------------------------------------------------------#
# END OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#

function (define_library name source include_dirs link_libraries compile_defs)
  add_library(${name} SHARED ${source})
  target_include_directories(${name} PRIVATE ${include_dirs})
  target_link_libraries(${name} ${link_libraries})
  target_compile_definitions(${name} PRIVATE ${compile_defs})
  set_target_properties(${name} PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
  endif(MSVC)

  install(
    TARGETS ${name}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib  # For Windows
    )
endfunction()


define_library(
  libpaddleaudio
  "${LIBPADDLEAUDIO_SOURCES}"
  "${LIBPADDLEAUDIO_INCLUDE_DIRS}"
  "${LIBPADDLEAUDIO_LINK_LIBRARIES}"
  "${LIBPADDLEAUDIO_COMPILE_DEFINITIONS}"
)

if (APPLE)
  add_custom_command(TARGET libpaddleaudio POST_BUILD COMMAND install_name_tool -change "${GFORTRAN_LIBRARIES_DIR}/libgcc_s.1.1.dylib" "@loader_path/libgcc_s.1.1.dylib" libpaddleaudio.so)
endif(APPLE)

#if (UNIX AND NOT APPLE)
#  set_target_properties(libpaddleaudio PROPERTIES INSTALL_RPATH "$ORIGIN")
#endif()

if (APPLE)
  set(AUDIO_LIBRARY libpaddleaudio CACHE INTERNAL "")
else()
  set(AUDIO_LIBRARY -Wl,--no-as-needed libpaddleaudio -Wl,--as-needed CACHE INTERNAL "")
endif()

  ################################################################################
# _paddleaudio.so
################################################################################
if (BUILD_PADDLEAUDIO_PYTHON_EXTENSION)
if (WIN32)
  find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)
  set(ADDITIONAL_ITEMS Python3::Python)
endif()
function(define_extension name sources include_dirs libraries definitions)
  add_library(${name} SHARED ${sources})
  target_compile_definitions(${name} PRIVATE "${definitions}")
  target_include_directories(
    ${name} PRIVATE ${PROJECT_SOURCE_DIR} ${Python_INCLUDE_DIR} ${pybind11_INCLUDE_DIR} ${include_dirs})
  target_link_libraries(
    ${name}
    ${libraries}
    ${PYTHON_LIBRARY}
    ${ADDITIONAL_ITEMS}
    )
  set_target_properties(${name} PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
  endif(MSVC)
  if (APPLE)
    # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
    # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
    set_target_properties(${name} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()
  install(
    TARGETS ${name}
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .  # For Windows
    )
endfunction()

set(
  EXTENSION_SOURCES
  pybind/pybind.cpp
  )
#----------------------------------------------------------------------------#
# START OF CUSTOMIZATION LOGICS
#----------------------------------------------------------------------------#
if(BUILD_SOX)
  list(
    APPEND
    EXTENSION_SOURCES
    pybind/sox/effects.cpp
    pybind/sox/effects_chain.cpp
    pybind/sox/io.cpp
    pybind/sox/types.cpp
    pybind/sox/utils.cpp
    )
endif()

if(BUILD_KALDI)
  list(
    APPEND
    EXTENSION_SOURCES
    pybind/kaldi/kaldi_feature_wrapper.cc
    pybind/kaldi/kaldi_feature.cc
    )
endif()
#----------------------------------------------------------------------------#
# END OF CUSTOMIZATION LOGICS
#----------------------------------------------------------------------------#
define_extension(
  _paddleaudio
  "${EXTENSION_SOURCES}"
  ""
  libpaddleaudio
  "${LIBPADDLEAUDIO_COMPILE_DEFINITIONS}"
  )
# if(BUILD_CTC_DECODER)
#   set(
#     DECODER_EXTENSION_SOURCES
#     decoder/bindings/pybind.cpp
#     )
#   define_extension(
#     _paddleaudio_decoder
#     "${DECODER_EXTENSION_SOURCES}"
#     ""
#     "libpaddleaudio_decoder"
#     "${LIBPADDLEAUDIO_DECODER_DEFINITIONS}"
#     )
# endif()
# if(USE_FFMPEG)
#   set(
#     FFMPEG_EXTENSION_SOURCES
#     ffmpeg/pybind/typedefs.cpp
#     ffmpeg/pybind/pybind.cpp
#     ffmpeg/pybind/stream_reader.cpp
#     )
#   define_extension(
#     _paddleaudio_ffmpeg
#     "${FFMPEG_EXTENSION_SOURCES}"
#     "${FFMPEG_INCLUDE_DIRS}"
#     "libpaddleaudio_ffmpeg"
#     "${LIBPADDLEAUDIO_DECODER_DEFINITIONS}"
#     )
# endif()
endif()

if (APPLE)
  add_custom_command(TARGET _paddleaudio POST_BUILD COMMAND install_name_tool -change "${GFORTRAN_LIBRARIES_DIR}/libgcc_s.1.1.dylib" "@loader_path/lib/libgcc_s.1.1.dylib" _paddleaudio.so)
endif(APPLE)

#if (UNIX AND NOT APPLE)
#  set_target_properties(_paddleaudio PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
#endif()
