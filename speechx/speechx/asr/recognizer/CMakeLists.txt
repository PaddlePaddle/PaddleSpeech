set(srcs)

if (USING_DS2)
list(APPEND srcs
recognizer.cc
)
endif()

if (USING_U2)
  list(APPEND srcs
    u2_recognizer.cc
  )
endif()

add_library(recognizer STATIC ${srcs})
target_link_libraries(recognizer PUBLIC decoder)

# test
if (USING_DS2)
  set(BINS recognizer_main)

  foreach(bin_name IN LISTS BINS)
    add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
    target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
    target_link_libraries(${bin_name} PUBLIC recognizer nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util ${DEPS})
  endforeach()
endif()


if (USING_U2)
  set(TEST_BINS 
    u2_recognizer_main
    u2_recognizer_thread_main
  )

  foreach(bin_name IN LISTS TEST_BINS)
    add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
    target_include_directories(${bin_name} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
    target_link_libraries(${bin_name} recognizer nnet decoder fst utils gflags glog kaldi-base kaldi-matrix kaldi-util)
    target_compile_options(${bin_name}  PRIVATE ${PADDLE_COMPILE_FLAGS})
    target_include_directories(${bin_name}  PRIVATE ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
    target_link_libraries(${bin_name}  ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})
  endforeach()

endif()

