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


# default this script executable all data prepare process
# stage and stop_stage can control the process range between stage and stop_stage
stage=-1        
stop_stage=100

# config_conf contain the feature and augmentation config
# default config conf is ./conf/default.yaml
# generally, we set all the parameters in the config file, containing the training parameters
config_conf=""

# default, use cmvn feature
cmvn=true             

. parse_options.sh || exit -1;

src=$1
dir=$2

# this script create the data/{train,dev} directory for train and dev data from voxceleb1
mkdir -p data/{train,dev}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Generate the dataset description file to data/train and data/dev directory ..."
    python3 ./local/voxceleb1.py  \
            --data-dir ${src} \
            --target-dir ${dir}
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Compute the dataset feature according the config: ${config_conf}"
    python3 ./local/compute_feature.py --config ${config_conf} \
            --dataset ./data/manifest_dev.json --feat ./data/manifest_dev.feat

    python3 ./local/compute_feature.py --config ${config_conf} \
            --dataset ./data/manifest_train.json --feat ./data/manifest_train.feat        
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "apply the cmvn"
    python3 ./local/apply_cmvn.py --cmvn data/cmvn.feat \
            --feat ./data/manifest_dev.feat --target ./data/apply_cmvnmanifest_dev.feat
    
    python3 ./local/apply_cmvn.py --cmvn data/cmvn.feat \
            --feat ./data/manifest_train.feat --target ./data/apply_cmvnmanifest_train.feat
fi

