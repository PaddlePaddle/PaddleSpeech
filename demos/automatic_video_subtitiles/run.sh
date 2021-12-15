#!/bin/bash

video_url=https://paddlespeech.bj.bcebos.com/demos/asr_demos/subtitle_demo1.mp4
video_file=$(basename ${video_url})
audio_file=$(echo ${video_file} | awk -F'.' '{print $1}').wav
num_channels=1
sr=16000

# Download video
if [ ! -f ${video_file} ]; then
    wget -c ${video_url}
fi

# Extract audio from video
if [ ! -f ${audio_file} ]; then
    ffmpeg -i ${video_file} -ac ${num_channels} -ar ${sr} -vn ${audio_file}
fi

python -u recognize.py --input ${audio_file}
exit 0
