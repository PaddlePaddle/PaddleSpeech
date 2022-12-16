include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/base
)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/../
${CMAKE_CURRENT_SOURCE_DIR}/utils
)
add_subdirectory(utils)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/frontend
)
add_subdirectory(frontend)
