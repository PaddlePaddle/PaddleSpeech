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
add_definitions(common base utils kaldi-matrix frontend)