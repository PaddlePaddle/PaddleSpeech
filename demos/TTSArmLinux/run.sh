#!/bin/bash
set -e

cd "$(dirname "$(realpath "$0")")"

# load configure
. ./config.sh

# create dir
mkdir -p "$OUTPUT_DIR"

# run
set -x
./build/paddlespeech_tts_demo \
    --front_conf "$FRONT_CONF" \
    --acoustic_model "$ACOUSTIC_MODEL_PATH" \
    --vocoder "$VOCODER_PATH" \
    "$@"
# end
