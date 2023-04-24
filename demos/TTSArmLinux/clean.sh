#!/bin/bash
set -e
set -x

cd "$(dirname "$(realpath "$0")")"

BASE_DIR="$PWD"

# load configure
. ./config.sh

# remove dirs
set -x

rm -rf "$OUTPUT_DIR"
rm -rf "$LIBS_DIR"
rm -rf "$MODELS_DIR"
rm -rf "$BASE_DIR/build"

"$BASE_DIR/src/TTSCppFrontend/clean.sh"

# 符号连接
rm "$BASE_DIR/dict"
