cmake_minimum_required(VERSION 3.10)
project(paddlespeech_tts_demo)


########## Global Options ##########

option(WITH_FRONT_DEMO "Build front demo" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ABSL_PROPAGATE_CXX_STD ON)


########## ARM Options ##########

set(CMAKE_SYSTEM_NAME Linux)
if(ARM_ABI STREQUAL "armv8")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    #set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    #set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
elseif(ARM_ABI STREQUAL "armv7hf")
    set(CMAKE_SYSTEM_PROCESSOR arm)
    #set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
    #set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
else()
    message(FATAL_ERROR "Unknown arch abi ${ARM_ABI}, only support armv8 and armv7hf.")
    return()
endif()


########## Paddle Lite Options ##########

message(STATUS "TARGET ARCH ABI: ${ARM_ABI}")
message(STATUS "PADDLE LITE DIR: ${PADDLE_LITE_DIR}")

include_directories(${PADDLE_LITE_DIR}/include)
link_directories(${PADDLE_LITE_DIR}/libs/${ARM_ABI})
link_directories(${PADDLE_LITE_DIR}/lib)

if(ARM_ABI STREQUAL "armv8")
    set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
elseif(ARM_ABI STREQUAL "armv7hf")
    set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
endif()


########## Dependencies ##########

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP C flags:  ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP CXX flags:  ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP OpenMP_CXX_LIB_NAMES:  ${OpenMP_CXX_LIB_NAMES}")
    message(STATUS "OpenMP OpenMP_CXX_LIBRARIES:  ${OpenMP_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not found OpenMP!")
    return()
endif()


############### tts cpp frontend ###############

add_subdirectory(TTSCppFrontend)

include_directories(
    TTSCppFrontend/src
    third-party/build/src/cppjieba/include
    third-party/build/src/limonp/include
)


############### paddlespeech_tts_demo ###############

add_executable(paddlespeech_tts_demo main.cc)
target_link_libraries(paddlespeech_tts_demo paddle_light_api_shared paddlespeech_tts_front)
