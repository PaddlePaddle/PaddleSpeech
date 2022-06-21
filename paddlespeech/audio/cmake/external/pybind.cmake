include(FetchContent)
FetchContent_Declare(
  pybind
  URL      https://github.com/pybind/pybind11/archive/refs/tags/v2.9.0.zip 
  URL_HASH SHA256=1c6e0141f7092867c5bf388bc3acdb2689ed49f59c3977651394c6c87ae88232
)
FetchContent_MakeAvailable(pybind)
include_directories(${pybind_SOURCE_DIR}/include)

