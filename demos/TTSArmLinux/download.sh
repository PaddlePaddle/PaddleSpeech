#!/bin/bash
set -e

cd "$(dirname "$(realpath "$0")")"
basedir="$PWD"

mkdir -p ./libs ./models

download() {
    file="$1"
    url="$2"
    dir="$3"

    cd "$dir"
    echo "Downloading $file..."
    wget -O "$file" "$url"
    echo "Extracting $file..."
    tar -vxf "$file"
}

download 'inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz' \
    'https://github.com/SwimmingTiger/Paddle-Lite/releases/download/68b66fd35/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz' \
    "$basedir/libs"

download 'fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz' \
    'https://paddlespeech.bj.bcebos.com/demos/TTSAndroid/fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz' \
    "$basedir/models"
