#!/bin/bash
set -e

cd "$(dirname "$(realpath "$0")")"

# load configure
. ./config.sh

# create dir
rm -rf ./output
mkdir -p ./output

# run
for i in {1..10}; do
    (set -x; ./build/paddlespeech_tts_demo "$AM_MODEL_PATH" "$VOC_MODEL_PATH" $i ./output/$i.wav)
done

ls -lh "$PWD"/output/*.wav
