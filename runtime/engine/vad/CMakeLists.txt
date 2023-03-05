

set(bin_name silero_vad_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc vad.cc)
target_link_libraries(${bin_name} ${FASTDEPLOY_LIBS} gflags extern_glog)
