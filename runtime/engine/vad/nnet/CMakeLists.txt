set(srcs 
    vad.cc 
)

add_library(pps_vad ${srcs})
target_link_libraries(pps_vad PUBLIC ${FASTDEPLOY_LIBS} common)


set(bin_name vad_nnet_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
# target_link_libraries(${bin_name} ${FASTDEPLOY_LIBS} vad gflags extern_glog)
target_link_libraries(${bin_name} ${FASTDEPLOY_LIBS} pps_vad)


file(RELATIVE_PATH DEST_DIR ${ENGINE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS pps_vad DESTINATION lib)