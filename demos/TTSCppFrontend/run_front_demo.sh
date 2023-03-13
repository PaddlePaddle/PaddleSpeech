#!/bin/bash
set -e
set -x

cd "$(dirname "$(realpath "$0")")"

./build/tts_front_demo "$@"
