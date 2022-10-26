include(FetchContent)


set(BUILD_SHARED_LIBS OFF) # up to you
set(BUILD_TESTING OFF) # to disable abseil test, or gtest will fail.
set(ABSL_ENABLE_INSTALL ON) # now you can enable install rules even in subproject...

FetchContent_Declare(
  absl
  GIT_REPOSITORY "https://github.com/abseil/abseil-cpp.git"
  GIT_TAG "20210324.1"
)
FetchContent_MakeAvailable(absl)

set(EIGEN3_INCLUDE_DIR ${Eigen3_SOURCE_DIR})
include_directories(${absl_SOURCE_DIR})