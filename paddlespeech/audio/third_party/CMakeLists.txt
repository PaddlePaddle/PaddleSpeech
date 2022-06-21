set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

add_library(libsox INTERFACE)
if (BUILD_SOX)
  add_subdirectory(sox)
  target_include_directories(libsox INTERFACE ${SOX_INCLUDE_DIR})
  target_link_libraries(libsox INTERFACE ${SOX_LIBRARIES})
endif()

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/kaldi
)
add_subdirectory(kaldi)
