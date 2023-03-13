include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/../
)
add_subdirectory(base)
add_subdirectory(utils)
add_subdirectory(matrix)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/frontend
)
add_subdirectory(frontend)

add_library(common INTERFACE)
target_link_libraries(common  INTERFACE base utils kaldi-matrix frontend)
install(TARGETS base DESTINATION lib)
install(TARGETS utils DESTINATION lib)
install(TARGETS kaldi-matrix DESTINATION lib)
install(TARGETS frontend DESTINATION lib)