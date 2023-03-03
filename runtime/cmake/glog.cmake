include(FetchContent)

if(ANDROID)
else() # UNIX
  add_definitions(-DWITH_GLOG)
  FetchContent_Declare(
    extern_glog
    URL      https://paddleaudio.bj.bcebos.com/build/glog-0.4.0.zip
    URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
  )
  FetchContent_MakeAvailable(extern_glog)
  include_directories(${glog_BINARY_DIR} ${glog_SOURCE_DIR}/src)
endif()

add_library(glog INTERFACE)

if(ANDROID)
else() # UNIX
  add_dependencies(gtest extern_glog)
endif()