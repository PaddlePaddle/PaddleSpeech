#!/bin/bash
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

. ./path.sh

stage=0
stop_stage=100
exp_dir=exp/ecapa-tdnn-vox12-big/            # experiment directory
conf_path=conf/ecapa_tdnn.yaml
audio_path="demo/voxceleb/00001.wav"
use_gpu=true

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;

if [ $# -ne 0 ] ; then
   echo "Usage: $0 [options]";
   echo "e.g.: $0 ./data/ exp/voxceleb12/ conf/ecapa_tdnn.yaml"
   echo "Options: "
   echo "  --use-gpu <true,false|true>      # specify is gpu is to be used for training"
   echo "  --stage <stage|-1>               # Used to run a partially-completed data process from somewhere in the middle."
   echo "  --stop-stage <stop-stage|100>    # Used to run a partially-completed data process stop stage in the middle"
   echo "  --exp-dir                        # experiment directorh, where is has the model.pdparams"
   echo "  --conf-path                      # configuration file for extracting the embedding"
   echo "  --audio-path                     # audio-path, which will be processed to extract the embedding"
   exit 1;
fi

# set the test device
device="cpu"
if ${use_gpu}; then
    device="gpu"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # extract the audio embedding
    python3 ${BIN_DIR}/extract_emb.py --device ${device} \
            --config ${conf_path} \
            --audio-path ${audio_path} --load-checkpoint ${exp_dir}
fi