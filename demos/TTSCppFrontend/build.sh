#!/bin/bash
set -e
set -x

cd "$(dirname "$(realpath "$0")")"

echo "************* Download & Build Dependencies *************"
./build-depends.sh "$@"

echo "************* Build Front Lib and Demo *************"
mkdir -p ./build
cd ./build
cmake ..

if [ "$*" = "" ]; then
    make -j$(nproc)
else
    make "$@"
fi

echo "Done."
