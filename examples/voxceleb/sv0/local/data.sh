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
stage=0
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

# Generally the `MAIN_ROOT` refers to the root of PaddleSpeech,
# which is defined in the path.sh
# And we will download the voxceleb data and rirs noise to ${MAIN_ROOT}/dataset
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   # download data, generate manifests
   # we will generate the manifest.{dev,test} file from ${TARGET_DIR}/voxceleb/vox1/{dev,test} directory
   # and generate the meta info and download the trial file
   # manifest.dev: 148642
   # manifest.test: 4847
   echo "Start to download vox1 dataset and generate the manifest files "
   python3 ${TARGET_DIR}/voxceleb/voxceleb1.py \
      --manifest_prefix="${dir}/vox1/manifest" \
      --target_dir="${TARGET_DIR}/voxceleb/vox1/"

   if [ $? -ne 0 ]; then
      echo "Prepare voxceleb1 failed. Terminated."
      exit 1
   fi

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   # download voxceleb2 data
   # we will download the data and unzip the package
   # and we will store the m4a file in ${TARGET_DIR}/voxceleb/vox2/{dev,test}
   echo "start to download vox2 dataset"
   python3 ${TARGET_DIR}/voxceleb/voxceleb2.py \
      --download \
      --target_dir="${TARGET_DIR}/voxceleb/vox2/"

   if [ $? -ne 0 ]; then
      echo "Download voxceleb2 dataset failed. Terminated."
      exit 1
   fi

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
   # convert the m4a to wav
   # and we will not delete the original m4a file
   echo "start to convert the m4a to wav"
   bash local/convert.sh ${TARGET_DIR}/voxceleb/vox2/test/ || exit 1;
   
   if [ $? -ne 0 ]; then
      echo "Convert voxceleb2 dataset from m4a to wav failed. Terminated."
      exit 1
   fi
   echo "m4a convert to wav operation finished"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
   # generate the vox2 manifest file from wav file
   # we will generate the ${dir}/vox2/manifest.vox2
   # because we use all the vox2 dataset to train, so collect all the vox2 data in one file
   echo "start generate the vox2 manifest files"
   python3 ${TARGET_DIR}/voxceleb/voxceleb2.py \
      --generate \
      --manifest_prefix="${dir}/vox2/manifest" \
      --target_dir="${TARGET_DIR}/voxceleb/vox2/"

   if [ $? -ne 0 ]; then
      echo "Prepare voxceleb2 dataset failed. Terminated."
      exit 1
   fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
   # generate the vox csv file
   # Currently, our training system use csv file for dataset
   echo "convert the json format to csv format to be compatible with training process"
   python3 local/make_vox_csv_dataset_from_json.py\
      --train "${dir}/vox1/manifest.dev" "${dir}/vox2/manifest.vox2"\
      --test "${dir}/vox1/manifest.test" \
      --target_dir "${dir}/vox/" \
      --config ${conf_path}

   if [ $? -ne 0 ]; then
      echo "Prepare voxceleb failed. Terminated."
      exit 1
   fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
   # generate the open rir noise manifest file
   echo "generate the open rir noise manifest file"
   python3 ${TARGET_DIR}/rir_noise/rir_noise.py\
      --manifest_prefix="${dir}/rir_noise/manifest" \
      --target_dir="${TARGET_DIR}/rir_noise/"

   if [ $? -ne 0 ]; then
      echo "Prepare rir_noise failed. Terminated."
      exit 1
   fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
   # generate the open rir noise manifest file
   echo "generate the open rir noise csv file"
   python3 local/make_rirs_noise_csv_dataset_from_json.py \
      --noise_dir="${TARGET_DIR}/rir_noise/" \
      --data_dir="${dir}/rir_noise/" \
      --config ${conf_path}

   if [ $? -ne 0 ]; then
      echo "Prepare rir_noise failed. Terminated."
      exit 1
   fi
fi
