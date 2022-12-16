cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(ASR LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/server)

add_subdirectory(decoder)
add_subdirectory(recognizer)
add_subdirectory(nnet)
add_subdirectory(server)
