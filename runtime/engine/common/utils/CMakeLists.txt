
add_library(utils
  file_utils.cc
  math.cc
  strings.cc
  audio_process.cc
)


if(WITH_TESTING)
  enable_testing()
  link_libraries(gtest_main gmock)

  add_executable(strings_test strings_test.cc)
  target_link_libraries(strings_test PUBLIC utils)
  add_test(
    NAME strings_test
    COMMAND strings_test
  )
endif()