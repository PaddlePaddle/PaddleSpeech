include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
  pybind
  URL      https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.zip
  URL_HASH SHA256=225df6e6dea7cea7c5754d4ed954e9ca7c43947b849b3795f87cb56437f1bd19
)
FetchContent_MakeAvailable(pybind)
include_directories(${pybind_SOURCE_DIR}/include)

