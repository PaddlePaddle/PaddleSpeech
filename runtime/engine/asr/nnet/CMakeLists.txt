set(srcs decodable.cc nnet_producer.cc)

list(APPEND srcs u2_nnet.cc)
if(WITH_ONNX)
    list(APPEND srcs u2_onnx_nnet.cc)
endif()
add_library(nnet STATIC ${srcs})
target_link_libraries(nnet utils)
if(WITH_ONNX)
    target_link_libraries(nnet ${FASTDEPLOY_LIBS})
endif()

target_compile_options(nnet  PUBLIC ${PADDLE_COMPILE_FLAGS})
target_include_directories(nnet  PUBLIC ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

# test bin  
#set(bin_name u2_nnet_main)
#add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
#target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
#target_link_libraries(${bin_name} utils kaldi-util kaldi-matrix gflags glog nnet)

#target_compile_options(${bin_name}  PRIVATE ${PADDLE_COMPILE_FLAGS})
#target_include_directories(${bin_name}  PRIVATE ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
#target_link_libraries(${bin_name}  ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})

set(bin_name u2_nnet_thread_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(${bin_name} utils kaldi-util kaldi-matrix gflags glog nnet frontend)

target_compile_options(${bin_name}  PRIVATE ${PADDLE_COMPILE_FLAGS})
target_include_directories(${bin_name}  PRIVATE ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(${bin_name}  ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})