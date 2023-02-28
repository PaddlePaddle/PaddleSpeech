# set(CMAKE_CXX_STANDARD 11)

# # 指定下载解压后的fastdeploy库路径
# set(FASTDEPLOY_INSTALL_DIR "fdlib/fastdeploy-linux-x64-1.0.4" CACHE STRING force)

# if(NOT EXISTS ${FASTDEPLOY_INSTALL_DIR})
#     message(FATAL_ERROR "Please using cmake -B build -DFASTDEPLOY_INSTALL_DIR=${FASTDEPLOY_INSTALL_DIR}")
# endif()

# include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# # 添加FastDeploy依赖头文件
# include_directories(${FASTDEPLOY_INCS})

add_executable(infer_onnx_silero_vad ${CMAKE_CURRENT_SOURCE_DIR}/infer_onnx_silero_vad.cc wav.h vad.cc vad.h)

# 添加FastDeploy库依赖
target_link_libraries(infer_onnx_silero_vad ${FASTDEPLOY_LIBS})
