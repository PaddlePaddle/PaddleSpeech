ANDROID_NDK=/workspace/zhanghui/android-sdk/android-ndk-r25c
ANDROID_TOOLCHAIN=clang
FASTDEPLOY_INSTALL_DIR=./fdlib/fastdeploy-android-1.0.3-shared/
TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake

cmake -B build  -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
         -DCMAKE_BUILD_TYPE=Release \
         -DANDROID_ABI="arm64-v8a" \
         -DANDROID_NDK=${ANDROID_NDK} \
         -DANDROID_PLATFORM="android-21" \
         -DANDROID_STL=c++_shared \
         -DANDROID_TOOLCHAIN=${ANDROID_TOOLCHAIN} \
         -DFASTDEPLOY_INSTALL_DIR=${FASTDEPLOY_INSTALL_DIR} \
         -Wno-dev
