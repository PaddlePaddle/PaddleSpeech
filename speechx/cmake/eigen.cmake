include(FetchContent)

# update eigen to the commit id f612df27 on 03/16/2021
set(EIGEN_PREFIX_DIR ${fc_patch}/eigen3)

FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  PREFIX            ${EIGEN_PREFIX_DIR}
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

set(EIGEN_BUILD_DOC OFF)
# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set( OFF)
FetchContent_MakeAvailable(Eigen3)

message(STATUS "eigen src dir: ${Eigen3_SOURCE_DIR}")
message(STATUS "eigen bin dir: ${Eigen3_BINARY_DIR}")
#include_directories(${Eigen3_SOURCE_DIR})
#link_directories(${Eigen3_BINARY_DIR})