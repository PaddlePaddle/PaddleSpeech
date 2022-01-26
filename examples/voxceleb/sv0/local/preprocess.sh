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
    python3 ./local/voxceleb1.py --data-dir ${src}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Compute the dataset feature according the config: ${config_conf}"

    for x in train dev; do
        python3 ./local/compute_feature.py --config ${config_conf} \
                --dataset ./data/${x}/data.json --feat ./data/${x}/feat.npz
    done      
fi

if $cmvn; then
    echo "the script use cmvn configuration"
    
    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        echo "compute the the cmvn from data/train/feat.npz"
        feat_dim=$(python3 local/compute_feat_dim.py  --feat ./data/train/feat.npz)
        python3 local/compute_cmvn.py \
                    --feat-dim ${feat_dim} \
                    --cmvn ./data/train/cmvn.npz \
                    --feat ./data/train/feat.npz

    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        echo "apply the cmvn to the data/train/feat.npz and data/dev/feat.npz"
        for x in train dev; do
            mkdir -p ${dir}/${x}
            python3 ./local/apply_cmvn.py \
                    --cmvn data/train/cmvn.npz \
                    --feat ./data/${x}/feat.npz \
                    --target ${dir}/${x}/feat.npz
        done
    fi

    cp data/train/cmvn.npz ${dir}
fi

