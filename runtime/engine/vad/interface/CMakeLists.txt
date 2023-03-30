set(srcs 
    vad_interface.cc 
)

add_library(pps_vad_interface SHARED ${srcs})
target_link_libraries(pps_vad_interface PUBLIC pps_vad extern_glog)


set(bin_name vad_interface_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_link_libraries(${bin_name} pps_vad_interface)
# set_target_properties(${bin_name} PROPERTIES PUBLIC_HEADER "vad_interface.h;../frontend/wav.h")

file(RELATIVE_PATH DEST_DIR ${ENGINE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS pps_vad_interface DESTINATION lib)
install(FILES vad_interface.h DESTINATION include/${DEST_DIR})

install(TARGETS vad_interface_main 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/${DEST_DIR}
)
install(FILES vad_interface_main.cc DESTINATION demo/${DEST_DIR})