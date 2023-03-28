# https://www.jianshu.com/p/33672fb819f5

PATH="/Applications/CMake.app/Contents/bin":"$PATH"
tools_dir=$1
ios_toolchain_cmake=${tools_dir}/"/ios-cmake-4.2.0/ios.toolchain.cmake"
fastdeploy_dir=${tools_dir}"/fastdeploy-ort-mac-build/"
build_targets=("OS64")
build_type_array=("Release")

#static_name="libocr"
#lib_name="libocr"

# Switch to workpath
current_path=`cd $(dirname $0);pwd`
work_path=${current_path}/
build_path=${current_path}/build/
output_path=${current_path}/output/
cd ${work_path}

# Clean
rm -rf ${build_path}
rm -rf ${output_path}

if [ "$1"x = "clean"x ]; then
    exit 0
fi

# Build Every Target
for target in "${build_targets[@]}"
do
    for build_type in "${build_type_array[@]}"
    do    
        echo -e "\033[1;36;40mBuilding ${build_type} ${target} ... \033[0m"
        target_build_path=${build_path}/${target}/${build_type}/
        mkdir -p ${target_build_path}

        cd ${target_build_path}
        if [ $? -ne 0 ];then
            echo -e "\033[1;31;40mcd ${target_build_path} failed \033[0m"
            exit -1
        fi
        
        if [ ${target} == "OS64" ];then
            fastdeploy_install_dir=${fastdeploy_dir}/arm64
	    else
            fastdeploy_install_dir=""
            echo "fastdeploy_install_dir is null"
            exit -1
	    fi

        cmake -DCMAKE_TOOLCHAIN_FILE=${ios_toolchain_cmake} \
            -DBUILD_IN_MACOS=ON \
            -DBUILD_SHARED_LIBS=OFF \
            -DWITH_ASR=OFF \
            -DWITH_CLS=OFF \
            -DWITH_VAD=ON \
	        -DFASTDEPLOY_INSTALL_DIR=${fastdeploy_install_dir} \
            -DPLATFORM=${target} ../../../

        cmake --build . --config ${build_type}

		mkdir output
        cp engine/vad/interface/libpps_vad_interface.a output
        cp engine/vad/interface/vad_interface_main.app/vad_interface_main output
        cp ${fastdeploy_install_dir}/lib/libfastdeploy.dylib output
	    cp ${fastdeploy_install_dir}/third_libs/install/onnxruntime/lib/libonnxruntime.dylib output	

    done
done

## combine all ios libraries
#DEVROOT=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/
#LIPO_TOOL=${DEVROOT}/usr/bin/lipo
#LIBRARY_PATH=${build_path}
#LIBRARY_OUTPUT_PATH=${output_path}/IOS
#mkdir -p ${LIBRARY_OUTPUT_PATH}
#
#${LIPO_TOOL}    \
#    -arch i386 ${LIBRARY_PATH}/ios_x86/Release/${lib_name}.a          \
#    -arch x86_64 ${LIBRARY_PATH}/ios_x86_64/Release/${lib_name}.a     \
#    -arch armv7 ${LIBRARY_PATH}/ios_armv7/Release/${lib_name}.a       \
#    -arch armv7s ${LIBRARY_PATH}/ios_armv7s/Release/${lib_name}.a     \
#    -arch arm64 ${LIBRARY_PATH}/ios_armv8/Release/${lib_name}.a       \
#    -output ${LIBRARY_OUTPUT_PATH}/${lib_name}.a -create
#
#cp ${work_path}/lib/houyi/lib/ios/libhouyi_score.a ${LIBRARY_OUTPUT_PATH}/
#cp ${work_path}/interface/ocr-interface.h ${output_path}
#cp ${work_path}/version/release.v ${output_path}
#
#echo -e "\033[1;36;40mBuild All Target Success At:\n${output_path}\033[0m"
#exit 0
