cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speechx LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kaldi)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(kaldi)
add_subdirectory(common)

if(WITH_ASR)
    add_subdirectory(asr)
endif()

if(WITH_CLS)
    add_subdirectory(cls)
endif()

if(WITH_VAD)
    add_subdirectory(vad)
endif()

add_subdirectory(codelab)