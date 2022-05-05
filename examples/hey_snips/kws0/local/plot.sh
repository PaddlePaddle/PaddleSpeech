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

if [ $# != 3 ];then
    echo "usage: ${0} config_path checkpoint output_file"
    exit -1
fi

keyword=$1
stats_file=$2
img_file=$3

python3 ${BIN_DIR}/plot_det_curve.py --keyword_label ${keyword} --stats_file ${stats_file} --img_file ${img_file}
