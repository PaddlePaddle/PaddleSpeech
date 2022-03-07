include(FetchContent)
FetchContent_Declare(
  kenlm
  GIT_REPOSITORY "https://github.com/kpu/kenlm.git"
  GIT_TAG "df2d717e95183f79a90b2fa6e4307083a351ca6a"
)
# https://github.com/kpu/kenlm/blob/master/cmake/modules/FindEigen3.cmake
set(EIGEN3_INCLUDE_DIR ${Eigen3_SOURCE_DIR})
FetchContent_MakeAvailable(kenlm)
include_directories(${kenlm_SOURCE_DIR})