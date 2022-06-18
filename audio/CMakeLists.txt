cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(paddlespeech)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_SOX "Build libsox statically" ON)

add_subdirectory(third_party)
add_subdirectory(audio/csrc)
