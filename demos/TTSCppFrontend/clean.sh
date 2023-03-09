#!/bin/bash
set -e
set -x

cd "$(dirname "$(realpath "$0")")"
rm -rf "./front_demo/dict"
rm -rf "./build"
rm -rf "./third-party/build"

echo "Done."
