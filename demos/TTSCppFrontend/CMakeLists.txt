cmake_minimum_required(VERSION 3.10)
project(paddlespeech_tts_cpp)


########## Global Options ##########

option(WITH_FRONT_DEMO "Build front demo" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ABSL_PROPAGATE_CXX_STD ON)


########## Dependencies ##########

set(ENV{PKG_CONFIG_PATH} "${CMAKE_SOURCE_DIR}/third-party/build/lib/pkgconfig")
find_package(PkgConfig REQUIRED)

# It is hard to load xxx-config.cmake in a custom location, so use pkgconfig instead.
pkg_check_modules(ABSL   REQUIRED absl_strings IMPORTED_TARGET)
pkg_check_modules(GFLAGS REQUIRED gflags       IMPORTED_TARGET)
pkg_check_modules(GLOG   REQUIRED libglog      IMPORTED_TARGET)

# load header-only libraries
include_directories(
    ${CMAKE_SOURCE_DIR}/third-party/build/src/cppjieba/include
    ${CMAKE_SOURCE_DIR}/third-party/build/src/limonp/include
)

find_package(Threads REQUIRED)


########## paddlespeech_tts_front ##########

include_directories(src)

file(GLOB FRONT_SOURCES
    ./src/base/*.cpp
    ./src/front/*.cpp
)
add_library(paddlespeech_tts_front STATIC ${FRONT_SOURCES})

target_link_libraries(
    paddlespeech_tts_front
    PUBLIC
    PkgConfig::GFLAGS
    PkgConfig::GLOG
    PkgConfig::ABSL
    Threads::Threads
)


########## tts_front_demo ##########

if (WITH_FRONT_DEMO)

    file(GLOB FRONT_DEMO_SOURCES front_demo/*.cpp)
    add_executable(tts_front_demo ${FRONT_DEMO_SOURCES})

    target_include_directories(tts_front_demo PRIVATE ./front_demo)
    target_link_libraries(tts_front_demo PRIVATE paddlespeech_tts_front)

endif (WITH_FRONT_DEMO)
