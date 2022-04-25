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

if [ $# != 2 ];then
    echo "usage: ${0} num_gpus config_path"
    exit -1
fi

ngpu=$1
cfg_path=$2

if [ ${ngpu} -gt 0 ]; then
    python3 -m paddle.distributed.launch --gpus $CUDA_VISIBLE_DEVICES ${BIN_DIR}/train.py \
    --config ${cfg_path}
else
    echo "set CUDA_VISIBLE_DEVICES to enable multi-gpus trainning."
    python3 ${BIN_DIR}/train.py \
    --config ${cfg_path}
fi
