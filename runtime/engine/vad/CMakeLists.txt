add_executable(infer_onnx_silero_vad ${CMAKE_CURRENT_SOURCE_DIR}/infer_onnx_silero_vad.cc wav.h vad.cc vad.h)

# 添加FastDeploy库依赖
target_link_libraries(infer_onnx_silero_vad ${FASTDEPLOY_LIBS} gflags glog)
