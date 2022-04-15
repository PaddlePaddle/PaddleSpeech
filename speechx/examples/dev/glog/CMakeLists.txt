cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

add_executable(glog_test ${CMAKE_CURRENT_SOURCE_DIR}/glog_test.cc)
target_link_libraries(glog_test glog)


add_executable(glog_logtostderr_test ${CMAKE_CURRENT_SOURCE_DIR}/glog_logtostderr_test.cc)
target_link_libraries(glog_logtostderr_test glog)