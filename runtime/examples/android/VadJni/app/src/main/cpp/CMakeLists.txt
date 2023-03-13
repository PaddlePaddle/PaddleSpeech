# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.

project("vadjni")


set(PPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${ANDROID_ABI})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        vadjni

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

message(STATUS "PPS_DIR=${PPS_DIR}")
target_link_libraries( # Specifies the target library.
        vadjni
        ${PPS_DIR}/libfastdeploy.so
        ${PPS_DIR}/libonnxruntime.so
        ${PPS_DIR}/libgflags_nothreads.a
        ${PPS_DIR}/libbase.a
        ${PPS_DIR}/libpps_vad.a
        ${PPS_DIR}/libpps_vad_interface.a
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})