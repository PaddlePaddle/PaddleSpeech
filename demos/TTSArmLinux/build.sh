#!/bin/bash
set -e

cd "$(dirname "$(realpath "$0")")"

# load configure
. ./config.sh

# build
echo "ARM_ABI is ${ARM_ABI}"
echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"

rm -rf build
mkdir -p build
cd build

cmake -DPADDLE_LITE_DIR="${PADDLE_LITE_DIR}" -DARM_ABI="${ARM_ABI}" ../src
make

echo "make successful!"
