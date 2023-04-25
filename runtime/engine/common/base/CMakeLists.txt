

if(WITH_ASR)
  add_compile_options(-DWITH_ASR)
  set(PPS_FLAGS_LIB "fst/flags.h")
else()
  set(PPS_FLAGS_LIB "gflags/gflags.h")
endif()

if(ANDROID)
  set(PPS_GLOG_LIB "base/log_impl.h")
else() #UNIX
  if(WITH_ASR)
    set(PPS_GLOG_LIB "fst/log.h")
  else()
    set(PPS_GLOG_LIB "glog/logging.h")
  endif()
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


if(ANDROID)
  set(csrc
    log_impl.cc
    glog_utils.cc
  )
  add_library(base ${csrc})
  target_link_libraries(base gflags)
else() # UNIX
  set(csrc)
  add_library(base INTERFACE)
endif()