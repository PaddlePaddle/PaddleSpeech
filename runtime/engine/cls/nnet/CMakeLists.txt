set(srcs 
    panns_nnet.cc 
    panns_interface.cc
)

add_library(cls ${srcs})
target_link_libraries(cls INTERFACE -static-libstdc++;-Wl,-Bsymbolic ${FASTDEPLOY_LIBS} kaldi-matrix kaldi-base frontend utils )

set(bin_name panns_nnet_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_link_libraries(${bin_name} -static-libstdc++;-Wl,-Bsymbolic gflags glog)
