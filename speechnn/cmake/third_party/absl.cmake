cmake_minimum_required(VERSION 3.14)
include(ExternalProject)
include(FetchContent)

FetchContent_Declare(
  absl
  GIT_REPOSITORY "https://github.com/abseil/abseil-cpp.git"
  GIT_TAG "20210324.1"
)

FetchContent_MakeAvailable(absl)


