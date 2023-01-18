include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

set(srcs cls_nnet.cc cls_interface.cc)

add_library(cls SHARED ${srcs})
target_link_libraries(cls -static-libstdc++;-Wl,-Bsymbolic ${FASTDEPLOY_LIBS} kaldi-native-fbank-core)

set(bin_name cls_nnet_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_link_libraries(${bin_name} -static-libstdc++;-Wl,-Bsymbolic cls)