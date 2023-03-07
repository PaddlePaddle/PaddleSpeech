#!/bin/bash
set -e

cd "$(dirname "$(realpath "$0")")"

# load configure
. ./config.sh

# remove dirs
set -x

rm -rf "$OUTPUT_DIR"
rm -rf "$LIBS_DIR"
rm -rf "$MODELS_DIR"
