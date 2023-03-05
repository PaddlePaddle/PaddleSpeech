cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

add_executable(glog_main ${CMAKE_CURRENT_SOURCE_DIR}/glog_main.cc)
target_link_libraries(glog_main extern_glog)


add_executable(glog_logtostderr_main ${CMAKE_CURRENT_SOURCE_DIR}/glog_logtostderr_main.cc)
target_link_libraries(glog_logtostderr_main extern_glog)
