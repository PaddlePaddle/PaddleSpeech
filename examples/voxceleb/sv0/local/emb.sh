#!/bin/bash
. ./path.sh

exp_dir=exp/ecapa-tdnn-vox12-big//epoch_10/            # experiment directory
conf_path=conf/ecapa_tdnn.yaml
audio_path="demo/voxceleb/00001.wav"

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

# extract the audio embedding
python3 ${BIN_DIR}/extract_emb.py --device "gpu" \
          --config ${conf_path} \
          --audio-path ${audio_path} --load-checkpoint ${exp_dir}