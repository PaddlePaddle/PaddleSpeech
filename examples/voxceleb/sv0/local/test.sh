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

stage=1
stop_stage=100
use_gpu=true    # if true, we run on GPU.

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;

if [ $# -ne 3 ] ; then
   echo "Usage: $0 [options] <data-dir> <exp-dir> <conf-path>";
   echo "e.g.: $0 ./data/ exp/voxceleb12/ conf/ecapa_tdnn.yaml"
   echo "Options: "
   echo "  --use-gpu <true,false|true>      # specify is gpu is to be used for training"
   echo "  --stage <stage|-1>               # Used to run a partially-completed data process from somewhere in the middle."
   echo "  --stop-stage <stop-stage|100>    # Used to run a partially-completed data process stop stage in the middle"
   exit 1;
fi

dir=$1
exp_dir=$2
conf_path=$3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   # test the model and compute the eer metrics
   python3 ${BIN_DIR}/test.py \
         --data-dir ${dir} \
         --load-checkpoint ${exp_dir} \
         --config ${conf_path}
fi
