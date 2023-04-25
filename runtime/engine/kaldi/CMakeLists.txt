include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(base)
add_subdirectory(util)
if(WITH_ASR)
    add_subdirectory(lat)
    add_subdirectory(fstext)
    add_subdirectory(decoder)
    add_subdirectory(lm)

    add_subdirectory(fstbin)
    add_subdirectory(lmbin)
endif()
