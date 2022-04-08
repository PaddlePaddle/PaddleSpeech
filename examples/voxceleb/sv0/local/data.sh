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

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;

if [ $# -ne 2 ] ; then
   echo "Usage: $0 [options] <data-dir> <conf-path>";
   echo "e.g.: $0 ./data/ conf/ecapa_tdnn.yaml"
   echo "Options: "
   echo "  --stage <stage|-1>               # Used to run a partially-completed data process from somewhere in the middle."
   echo "  --stop-stage <stop-stage|100>    # Used to run a partially-completed data process stop stage in the middle"
   exit 1;
fi

dir=$1
conf_path=$2
mkdir -p ${dir}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # data prepare for vox1 and vox2, vox2 must be converted from m4a to wav
    # we should use the local/convert.sh convert m4a to wav
    python3 local/data_prepare.py \
                        --data-dir ${dir} \
                        --config ${conf_path}
fi 

TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/voxceleb/voxceleb1.py \
      --manifest_prefix="data/vox1/manifest" \
      --target_dir="${TARGET_DIR}/voxceleb/vox1/"

    if [ $? -ne 0 ]; then
        echo "Prepare voxceleb failed. Terminated."
        exit 1
    fi

   #  for dataset in train dev test; do
   #      mv data/manifest.${dataset} data/manifest.${dataset}.raw
   #  done
fi