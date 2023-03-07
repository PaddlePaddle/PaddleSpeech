#!/bin/bash
set -e

cd "$(dirname "$(realpath "$0")")"

# load configure
. ./config.sh

# create dir
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# run
for i in {1..10}; do
    (set -x; ./build/paddlespeech_tts_demo "$AM_MODEL_PATH" "$VOC_MODEL_PATH" $i "$OUTPUT_DIR/$i.wav")
done

ls -lh "$OUTPUT_DIR"/*.wav
