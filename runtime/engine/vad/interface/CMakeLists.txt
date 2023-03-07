set(srcs 
    vad_interface.cc 
)

add_library(vad_interface ${srcs})
target_link_libraries(vad_interface INTERFACE ${FASTDEPLOY_LIBS} vad extern_glog)


set(bin_name vad_interface_main)
add_executable(${bin_name} ${CMAKE_CURRENT_SOURCE_DIR}/${bin_name}.cc)
target_link_libraries(${bin_name} ${FASTDEPLOY_LIBS} vad_interface)
# set_target_properties(${bin_name} PROPERTIES PUBLIC_HEADER "vad_interface.h;../frontend/wav.h")

install(TARGETS vad_interface DESTINATION lib)
install(FILES vad_interface.h DESTINATION include)
install(TARGETS vad_interface_main 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include
)
install(FILES vad_interface_main.cc DESTINATION demo)


# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME "paddlespeech_vad_interface")
set(CPACK_PACKAGE_VENDOR "paddlespeech")
set(CPACK_PACKAGE_VERSION_MAJOR 1)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_PACKAGE_DESCRIPTION "paddlespeech vad interface")
set(CPACK_PACKAGE_CONTACT "paddlespeech@baidu.com")
set(CPACK_SOURCE_GENERATOR "TGZ")
include (CPack)
