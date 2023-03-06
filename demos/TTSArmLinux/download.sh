#!/bin/bash
set -e

cd "$(dirname "$(realpath "$0")")"
basedir="$PWD"

mkdir -p ./libs ./models

download() {
    file="$1"
    url="$2"
    md5="$3"
    dir="$4"

    cd "$dir"

    if [ -f "$file" ] && [ "$(md5sum "$file" | awk '{ print $1 }')" = "$md5" ]; then
        echo "File $file (MD5: $md5) has been downloaded."
    else
        echo "Downloading $file..."
        wget -O "$file" "$url"

        # MD5 verify
        fileMd5="$(md5sum "$file" | awk '{ print $1 }')"
        if [ "$fileMd5" == "$md5" ]; then
            echo "File $file (MD5: $md5) has been downloaded."
        else
            echo "MD5 mismatch, file may be corrupt"
            echo "$file MD5: $fileMd5, it should be $md5"
        fi
    fi

    echo "Extracting $file..."
    echo '-----------------------'
    tar -vxf "$file"
    echo '======================='
}

download 'inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz' \
    'https://paddlespeech.bj.bcebos.com/demos/TTSArmLinux/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz' \
    '39e0c6604f97c70f5d13c573d7e709b9' \
    "$basedir/libs"

download 'inference_lite_lib.armlinux.armv7hf.gcc.with_extra.with_cv.tar.gz' \
    'https://paddlespeech.bj.bcebos.com/demos/TTSArmLinux/inference_lite_lib.armlinux.armv7hf.gcc.with_extra.with_cv.tar.gz' \
    'f5ceb509f0b610dafb8379889c5f36f8' \
    "$basedir/libs"

download 'fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz' \
    'https://paddlespeech.bj.bcebos.com/demos/TTSAndroid/fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz' \
    '93ef17d44b498aff3bea93e2c5c09a1e' \
    "$basedir/models"

echo "Done."
