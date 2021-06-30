#!/bin/bash

set -e

curdir=$PWD

test -d GigaSpeech || git clone https://github.com/SpeechColab/GigaSpeech.git
cd GigaSpeech
source env_vars.sh
utils/gigaspeech_download.sh ${curdir}/
