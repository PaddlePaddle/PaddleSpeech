set(srcs 
    vad.cc 
)

add_library(vad ${srcs})
target_link_libraries(vad INTERFACE ${FASTDEPLOY_LIBS} common)
target_link_libraries(vad PRIVATE common)


set(bin_name vad_nnet_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
# target_link_libraries(${bin_name} ${FASTDEPLOY_LIBS} vad gflags extern_glog)
target_link_libraries(${bin_name} ${FASTDEPLOY_LIBS} vad)