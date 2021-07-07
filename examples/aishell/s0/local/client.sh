#!/bin/bash

source path.sh

# run on MacOS
# brew install portaudio
# pip install pyaudio
# pip install keyboard

# start demo client
python3 -u ${BIN_DIR}/deploy/client.py \
--host_ip="localhost" \
--host_port=8086 \

if [ $? -ne 0 ]; then
    echo "Failed in starting demo client!"
    exit 1
fi

exit 0
