include(FetchContent)

set(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD 1 # Wrap download in script to log output
    LOG_UPDATE 1 # Wrap update in script to log output
    LOG_PATCH 1
    LOG_CONFIGURE 1# Wrap configure in script to log output
    LOG_BUILD 1 # Wrap build in script to log output
    LOG_INSTALL 1
    LOG_TEST 1 # Wrap test in script to log output
    LOG_MERGED_STDOUTERR 1
    LOG_OUTPUT_ON_FAILURE 1
)

if(NOT FASTDEPLOY_INSTALL_DIR)
    if(ANDROID)
        FetchContent_Declare(
            fastdeploy
            URL      https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-1.0.4-shared.tgz
            URL_HASH MD5=2a15301158e9eb157a4f11283689e7ba
            ${EXTERNAL_PROJECT_LOG_ARGS}
        )
        add_definitions("-DUSE_PADDLE_LITE_BAKEND")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -mfloat-abi=softfp -mfpu=vfpv3 -mfpu=neon -fPIC -pie -fPIE")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g0 -O3 -mfloat-abi=softfp -mfpu=vfpv3 -mfpu=neon -fPIC -pie -fPIE")
    else() # Linux
        FetchContent_Declare(
            fastdeploy
            URL      https://paddlespeech.bj.bcebos.com/speechx/fastdeploy/fastdeploy-1.0.5-x86_64-onnx.tar.gz 
            URL_HASH MD5=33900d986ea71aa78635e52f0733227c
            ${EXTERNAL_PROJECT_LOG_ARGS}
        )
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -msse -msse2")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse -msse2 -mavx -O3")
    endif()

    FetchContent_MakeAvailable(fastdeploy)

    set(FASTDEPLOY_INSTALL_DIR ${fc_patch}/fastdeploy-src)
endif()

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# fix compiler flags conflict, since fastdeploy using c++11 for project
# this line must after `include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)`
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
        file(GLOB_RECURSE ALL_OPENCV_DYN_LIBS ${OpenCV_DIR}/../../${DYN_LIB_SUFFIX})
    endif()
   
    list(REMOVE_ITEM ALL_DEPS_DYN_LIBS ${ALL_OPENCV_DYN_LIBS})

    if(WIN32)
        file(GLOB OPENCV_DYN_LIBS ${OpenCV_DIR}/x64/vc15/bin/${DYN_LIB_SUFFIX})
        install(FILES ${OPENCV_DYN_LIBS} DESTINATION lib)
    elseif(ANDROID AND (NOT WITH_ANDROID_OPENCV_STATIC))
        file(GLOB OPENCV_DYN_LIBS ${OpenCV_NATIVE_DIR}/libs/${ANDROID_ABI}/${DYN_LIB_SUFFIX})
        install(FILES ${OPENCV_DYN_LIBS} DESTINATION lib)
    else() # linux/mac
        file(GLOB OPENCV_DYN_LIBS ${OpenCV_DIR}/lib/${DYN_LIB_SUFFIX})
        install(FILES ${OPENCV_DYN_LIBS} DESTINATION lib)
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
