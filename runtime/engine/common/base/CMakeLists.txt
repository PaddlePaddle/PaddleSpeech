if(WITH_ASR)
  add_compile_options(-DWITH_ASR)
  set(PPS_FLAGS_LIB "fst/flags.h")
  set(PPS_GLOB_LIB "fst/log.h")
else()
  set(PPS_FLAGS_LIB "gflags/gflags.h")
  set(PPS_GLOB_LIB "glog/logging.h")
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/flags.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/flags.h @ONLY
  )
message(STATUS "Generated ${CMAKE_CURRENT_SOURCE_DIR}/flags.h")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/log.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/log.h @ONLY
  )
message(STATUS "Generated ${CMAKE_CURRENT_SOURCE_DIR}/log.h")