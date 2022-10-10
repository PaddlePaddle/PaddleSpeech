set(srcs decodable.cc)

if(USING_DS2)
  list(APPEND srcs ds2_nnet.cc)
endif()

if(USING_U2)
  list(APPEND srcs u2_nnet.cc)
endif()

add_library(nnet STATIC ${srcs})
target_link_libraries(nnet absl::strings)

if(USING_U2)
  target_compile_options(nnet  PUBLIC ${PADDLE_COMPILE_FLAGS})
  target_include_directories(nnet  PUBLIC ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
  # target_link_libraries(nnet  ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})
endif()


if(USING_DS2)
  set(bin_name ds2_nnet_main)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  target_link_libraries(${bin_name} utils kaldi-util kaldi-matrix gflags glog nnet)

  target_link_libraries(${bin_name} ${DEPS})
endif()

# test bin
if(USING_U2)
  set(bin_name u2_nnet_main)
  add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
  target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
  target_link_libraries(${bin_name} utils kaldi-util kaldi-matrix gflags glog nnet)

  target_compile_options(${bin_name}  PRIVATE ${PADDLE_COMPILE_FLAGS})
  target_include_directories(${bin_name}  PRIVATE ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
  target_link_libraries(${bin_name}  ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})
endif()
