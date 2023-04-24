#!/bin/bash
set -e
set -x

cd "$(dirname "$(realpath "$0")")"

cd ./third-party

mkdir -p build
cd build

cmake ..

if [ "$*" = "" ]; then
    make -j$(nproc)
else
    make "$@"
fi

echo "Done."
