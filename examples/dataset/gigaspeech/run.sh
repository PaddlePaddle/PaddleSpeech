#!/bin/bash

set -e

curdir=$PWD

test -d GigaSpeech || git clone https://github.com/SpeechColab/GigaSpeech.git


pushd GigaSpeech
source env_vars.sh
./utils/download_gigaspeech.sh ${curdir}/
#toolkits/kaldi/gigaspeech_data_prep.sh --train-subset XL /disk1/audio_data/gigaspeech ../data
popd
