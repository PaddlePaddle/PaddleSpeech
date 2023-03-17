include(FetchContent)

if(ANDROID)
else() # UNIX
  add_definitions(-DWITH_GLOG)
  FetchContent_Declare(
    glog
    URL      https://paddleaudio.bj.bcebos.com/build/glog-0.4.0.zip
    URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${GLOG_CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DWITH_GFLAGS=OFF
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
  )
  FetchContent_MakeAvailable(glog)
  include_directories(${glog_BINARY_DIR} ${glog_SOURCE_DIR}/src)
endif()


if(ANDROID)
  add_library(extern_glog INTERFACE)
  add_dependencies(glog gflags)
else() # UNIX
  add_library(extern_glog ALIAS glog)
  add_dependencies(glog gflags)
endif()