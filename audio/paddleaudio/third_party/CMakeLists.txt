set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

################################################################################
# sox
################################################################################
if (BUILD_SOX)
  add_subdirectory(sox)
endif()

################################################################################
# kaldi
################################################################################
if (BUILD_KALDI)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(kaldi-native-fbank/csrc)
endif()
