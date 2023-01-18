cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speechx LANGUAGES CXX)

add_compile_options(-fPIC)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kaldi)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(asr)
add_subdirectory(common)
add_subdirectory(kaldi)
add_subdirectory(codelab)
add_subdirectory(cls)