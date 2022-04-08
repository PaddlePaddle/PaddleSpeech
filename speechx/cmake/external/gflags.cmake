include(FetchContent)

FetchContent_Declare(
  gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.1.zip
  URL_HASH SHA256=4e44b69e709c826734dbbbd5208f61888a2faf63f239d73d8ba0011b2dccc97a
)

FetchContent_MakeAvailable(gflags)

# openfst need
include_directories(${gflags_BINARY_DIR}/include)