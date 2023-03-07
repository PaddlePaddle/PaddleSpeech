set(srcs 
    vad_interface.cc 
)

add_library(vad_interface ${srcs})
target_link_libraries(vad_interface INTERFACE ${FASTDEPLOY_LIBS} vad extern_glog)


set(bin_name vad_interface_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_link_libraries(${bin_name} ${FASTDEPLOY_LIBS} vad_interface)

install(TARGETS vad_interface DESTINATION lib)
install(FILES vad_interface.h DESTINATION include)
install(TARGETS vad_interface_main DESTINATION bin)