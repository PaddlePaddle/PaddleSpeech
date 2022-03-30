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
stage=5
stop_stage=5

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

# Generally the `MAIN_ROOT` refers to the root of PaddleSpeech,
# which is defined in the path.sh
# And we will download the 
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   # download data, generate manifests
   echo "Start to download vox1 dataset and generate the manifest files "
    python3 ${TARGET_DIR}/voxceleb/voxceleb1.py \
      --manifest_prefix="data/vox1/manifest" \
      --target_dir="${TARGET_DIR}/voxceleb/vox1/"

   if [ $? -ne 0 ]; then
      echo "Prepare voxceleb failed. Terminated."
      exit 1
   fi

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   # download voxceleb2 data
   echo "start to download vox2 dataset"
   python3 ${TARGET_DIR}/voxceleb/voxceleb2.py \
      --download \
      --target_dir="${TARGET_DIR}/voxceleb/vox2/"

   if [ $? -ne 0 ]; then
      echo "Prepare voxceleb failed. Terminated."
      exit 1
   fi

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
   # convert the m4a to wav
   echo "start to convert the m4a to wav"
   bash local/convert.sh ${TARGET_DIR}/voxceleb/vox2/test/ || exit 1;
   echo "m4a convert to wav operation finished"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
   # generate the vox2 manifest file 
   echo "start generate the vox2 manifest files"
   python3 ${TARGET_DIR}/voxceleb/voxceleb2.py \
      --generate \
      --manifest_prefix="data/vox2/manifest" \
      --target_dir="${TARGET_DIR}/voxceleb/vox2/"

   if [ $? -ne 0 ]; then
      echo "Prepare voxceleb failed. Terminated."
      exit 1
   fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
   # generate the vox2 manifest file 
   echo "convert the json format to csv format to be compatible with training process"
   python3 local/make_csv_dataset_from_json.py\
      --train "data/vox1/manifest.dev" \
      --test "data/vox1/manifest.test" \
      --target_dir "data/vox/" \
      --config ${conf_path}

   if [ $? -ne 0 ]; then
      echo "Prepare voxceleb failed. Terminated."
      exit 1
   fi
fi





