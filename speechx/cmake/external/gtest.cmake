include(FetchContent)
FetchContent_Declare(
  gtest
  URL      https://github.com/google/googletest/archive/release-1.10.0.zip
  URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
)
FetchContent_MakeAvailable(gtest)

include_directories(${gtest_BINARY_DIR} ${gtest_SOURCE_DIR}/src)