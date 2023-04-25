
# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("cls")

include_directories(src/main/cpp/)
file(GLOB CPP_FILES "src/main/cpp/*.cpp")

# 添加so库存放位置
set(distribution_DIR ${CMAKE_SOURCE_DIR}/jniLibs)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${CMAKE_CURRENT_SOURCE_DIR}/native-lib.cpp )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        native-lib
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libc++_shared.so
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libcls.so
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libfastdeploy.so
        ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libonnxruntime.so
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        )
