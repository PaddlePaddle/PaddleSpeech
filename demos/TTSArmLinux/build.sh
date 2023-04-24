#!/bin/bash
set -e
set -x

cd "$(dirname "$(realpath "$0")")"

BASE_DIR="$PWD"

# load configure
. ./config.sh

# build
echo "ARM_ABI is ${ARM_ABI}"
echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"

echo "Build depends..."
./build-depends.sh "$@"

mkdir -p "$BASE_DIR/build"
cd "$BASE_DIR/build"
cmake -DPADDLE_LITE_DIR="${PADDLE_LITE_DIR}" -DARM_ABI="${ARM_ABI}" ../src

if [ "$*" = "" ]; then
    make -j$(nproc)
else
    make "$@"
fi

echo "make successful!"
