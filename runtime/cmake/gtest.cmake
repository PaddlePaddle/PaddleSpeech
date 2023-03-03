
include(FetchContent)

if(ANDROID)
else() # UNIX
  FetchContent_Declare(
    extern_gtest
    URL      https://paddleaudio.bj.bcebos.com/build/gtest-release-1.11.0.zip
    URL_HASH SHA256=353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a
  )
  FetchContent_MakeAvailable(extern_gtest)

  include_directories(${gtest_BINARY_DIR} ${gtest_SOURCE_DIR}/src)
endif()

add_library(gtest INTERFACE)

if(ANDROID)
else() # UNIX
  add_dependencies(gtest extern_gtest)
endif()

if(WITH_TESTING)
  enable_testing()
endif()
