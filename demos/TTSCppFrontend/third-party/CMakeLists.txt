cmake_minimum_required(VERSION 3.10)
project(tts_third_party_libs)

include(ExternalProject)

# gflags
ExternalProject_Add(gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG        v2.2.2
    PREFIX         ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_DIR    ${CMAKE_CURRENT_BINARY_DIR}
    CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                   -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                   -DBUILD_STATIC_LIBS=OFF
                   -DBUILD_SHARED_LIBS=ON
)

# glog
ExternalProject_Add(
    glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG        v0.6.0
    PREFIX         ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_DIR    ${CMAKE_CURRENT_BINARY_DIR}
    CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                   -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    DEPENDS        gflags
)

# abseil
ExternalProject_Add(
    abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG        20230125.1
    PREFIX         ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_DIR    ${CMAKE_CURRENT_BINARY_DIR}
    CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                   -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                   -DABSL_PROPAGATE_CXX_STD=ON
)

# cppjieba (header-only)
ExternalProject_Add(
    cppjieba
    GIT_REPOSITORY https://github.com/yanyiwu/cppjieba.git
    GIT_TAG        v5.0.3
    PREFIX         ${CMAKE_CURRENT_BINARY_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

# limonp (header-only)
ExternalProject_Add(
    limonp
    GIT_REPOSITORY https://github.com/yanyiwu/limonp.git
    GIT_TAG        v0.6.6
    PREFIX         ${CMAKE_CURRENT_BINARY_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)
