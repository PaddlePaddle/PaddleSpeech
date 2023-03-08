include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/../
)

add_subdirectory(nnet)

add_subdirectory(interface)

if(ANDROID)
    add_subdirectory(jni)
endif()