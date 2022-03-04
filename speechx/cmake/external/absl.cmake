include(FetchContent)

FetchContent_Declare(
  absl
  GIT_REPOSITORY "https://github.com/abseil/abseil-cpp.git"
  GIT_TAG "20210324.1"
)
FetchContent_MakeAvailable(absl)

set(EIGEN3_INCLUDE_DIR ${Eigen3_SOURCE_DIR})
include_directories(${absl_SOURCE_DIR})