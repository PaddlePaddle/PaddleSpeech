set(srcs 
    vad_interface.cc 
)

add_library(pps_vad_interface ${srcs})
target_link_libraries(pps_vad_interface PUBLIC pps_vad extern_glog)


set(bin_name vad_interface_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_link_libraries(${bin_name} pps_vad_interface)
