set(ARCH "mserver_x86_64" CACHE STRING "Target Architecture:
android_arm, android_armv7, android_armv8, android_x86, android_x86_64,
mserver_x86_64, ubuntu_x86_64, ios_armv7, ios_armv7s, ios_armv8, ios_x86_64, ios_x86,
windows_x86")

set(FASTDEPLOY_DIR ${CMAKE_SOURCE_DIR}/fc_patch/fastdeploy)
if(NOT EXISTS ${FASTDEPLOY_DIR}/fastdeploy-linux-x64-1.0.4.tgz)
    exec_program("mkdir -p ${FASTDEPLOY_DIR} &&
    wget -c https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-1.0.4.tgz -P ${FASTDEPLOY_DIR} &&
    tar xzvf ${FASTDEPLOY_DIR}/fastdeploy-linux-x64-1.0.4.tgz -C ${FASTDEPLOY_DIR} &&
    mv ${FASTDEPLOY_DIR}/fastdeploy-linux-x64-1.0.4 ${FASTDEPLOY_DIR}/linux-x64")
endif()

if(NOT EXISTS ${FASTDEPLOY_DIR}/fastdeploy-android-1.0.4-shared.tgz)
    exec_program("mkdir -p ${FASTDEPLOY_DIR} &&
    wget -c https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-1.0.4-shared.tgz -P ${FASTDEPLOY_DIR} &&
    tar xzvf ${FASTDEPLOY_DIR}/fastdeploy-android-1.0.4-shared.tgz -C ${FASTDEPLOY_DIR} &&
    mv ${FASTDEPLOY_DIR}/fastdeploy-android-1.0.4-shared ${FASTDEPLOY_DIR}/android-armv7v8")
endif()


if(ANDROID)
    if(NOT DEFINED FASTDEPLOY_INSTALL_DIR)
        set(FASTDEPLOY_INSTALL_DIR ${FASTDEPLOY_DIR}/android-armv7v8)
    endif()

    add_definitions("-DUSE_PADDLE_LITE_BAKEND")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -mfloat-abi=softfp -mfpu=vfpv3 -mfpu=neon -fPIC -pie -fPIE")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g0 -O3 -mfloat-abi=softfp -mfpu=vfpv3 -mfpu=neon -fPIC -pie -fPIE")
elseif(UNIX)
    if(NOT DEFINED FASTDEPLOY_INSTALL_DIR)
        set(FASTDEPLOY_INSTALL_DIR ${FASTDEPLOY_DIR}/linux-x64)
    endif()

    add_definitions("-DUSE_PADDLE_INFERENCE_BACKEND")
    # add_definitions("-DUSE_ORT_BACKEND")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -msse -msse2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse -msse2 -mavx -O3")
endif()

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# fix compiler flags conflict, since fastdeploy using c++11 for project
set(CMAKE_CXX_STANDARD ${PPS_CXX_STANDARD})

include_directories(${FASTDEPLOY_INCS})

# install fastdeploy and dependents lib
# install_fastdeploy_libraries(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})


# No dynamic libs need to install while using
# FastDeploy static lib.
if(ANDROID AND WITH_ANDROID_STATIC_LIB)
    return()
endif()

set(DYN_LIB_SUFFIX "*.so*")
if(WIN32)
    set(DYN_LIB_SUFFIX "*.dll")
elseif(APPLE)
    set(DYN_LIB_SUFFIX "*.dylib*")
endif()

if(FastDeploy_DIR)
    set(DYN_SEARCH_DIR ${FastDeploy_DIR})
elseif(FASTDEPLOY_INSTALL_DIR)
    set(DYN_SEARCH_DIR ${FASTDEPLOY_INSTALL_DIR})
else()
    message(FATAL_ERROR "Please set FastDeploy_DIR/FASTDEPLOY_INSTALL_DIR before call install_fastdeploy_libraries.")
endif()

file(GLOB_RECURSE ALL_NEED_DYN_LIBS ${DYN_SEARCH_DIR}/lib/${DYN_LIB_SUFFIX})
file(GLOB_RECURSE ALL_DEPS_DYN_LIBS ${DYN_SEARCH_DIR}/third_libs/${DYN_LIB_SUFFIX})

if(ENABLE_VISION)
    # OpenCV
    if(ANDROID)
        file(GLOB_RECURSE ALL_OPENCV_DYN_LIBS ${OpenCV_NATIVE_DIR}/libs/${DYN_LIB_SUFFIX})
    else()
        file(GLOB_RECURSE ALL_OPENCV_DYN_LIBS ${OpenCV_DIR}/${DYN_LIB_SUFFIX})
    endif()
    list(REMOVE_ITEM ALL_DEPS_DYN_LIBS ${ALL_OPENCV_DYN_LIBS})

    if(WIN32)
        file(GLOB OPENCV_DYN_LIBS ${OpenCV_DIR}/x64/vc15/bin/${DYN_LIB_SUFFIX})
        install(FILES ${OPENCV_DYN_LIBS} DESTINATION lib})
    elseif(ANDROID AND (NOT WITH_ANDROID_OPENCV_STATIC))
        file(GLOB OPENCV_DYN_LIBS ${OpenCV_NATIVE_DIR}/libs/${ANDROID_ABI}/${DYN_LIB_SUFFIX})
        install(FILES ${OPENCV_DYN_LIBS} DESTINATION lib})
    else() # linux/mac
        file(GLOB OPENCV_DYN_LIBS ${OpenCV_DIR}/lib/${DYN_LIB_SUFFIX})
        install(FILES ${OPENCV_DYN_LIBS} DESTINATION lib})
    endif()

    # FlyCV
    if(ENABLE_FLYCV)
        file(GLOB_RECURSE ALL_FLYCV_DYN_LIBS ${FLYCV_LIB_DIR}/${DYN_LIB_SUFFIX})
        list(REMOVE_ITEM ALL_DEPS_DYN_LIBS ${ALL_FLYCV_DYN_LIBS})
        if(ANDROID AND (NOT WITH_ANDROID_FLYCV_STATIC))
        install(FILES ${ALL_FLYCV_DYN_LIBS} DESTINATION lib)
        endif()
    endif()
endif()

if(ENABLE_OPENVINO_BACKEND)
    # need plugins.xml for openvino backend
    set(OPENVINO_RUNTIME_BIN_DIR ${OPENVINO_DIR}/bin)
    file(GLOB OPENVINO_PLUGIN_XML ${OPENVINO_RUNTIME_BIN_DIR}/*.xml)
    install(FILES ${OPENVINO_PLUGIN_XML} DESTINATION lib)
endif()

# Install other libraries
install(FILES ${ALL_NEED_DYN_LIBS} DESTINATION lib)
install(FILES ${ALL_DEPS_DYN_LIBS} DESTINATION lib)