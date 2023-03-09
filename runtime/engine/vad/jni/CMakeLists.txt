set(srcs 
    vad_jni_interface.cc 
)

add_library(pps_vad_jni_interface ${srcs})
target_link_libraries(pps_vad_jni_interface PUBLIC pps_vad_interface)


file(RELATIVE_PATH DEST_DIR ${ENGINE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS pps_vad_jni_interface DESTINATION lib)
install(FILES vad_jni_interface.h DESTINATION include/${DEST_DIR})