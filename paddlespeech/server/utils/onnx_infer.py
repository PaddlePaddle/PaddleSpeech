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
import os
from typing import Optional

import onnxruntime as ort

from paddlespeech.cli.log import logger


def get_sess(model_path: Optional[os.PathLike] = None, sess_conf: dict = None):
    logger.debug(f"ort sessconf: {sess_conf}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if sess_conf.get('graph_optimization_level', 99) == 0:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # "gpu:0"
    providers = ['CPUExecutionProvider']
    if "gpu" in sess_conf.get("device", ""):
        device_id = int(sess_conf["device"].split(":")[1])
        providers = [('CUDAExecutionProvider', {'device_id': device_id})]

        # fastspeech2/mb_melgan can't use trt now!
        if sess_conf.get("use_trt", 0):
            providers = ['TensorrtExecutionProvider']
    logger.debug(f"ort providers: {providers}")

    if 'cpu_threads' in sess_conf:
        sess_options.intra_op_num_threads = sess_conf.get("cpu_threads", 0)
    else:
        sess_options.intra_op_num_threads = sess_conf.get(
            "intra_op_num_threads", 0)

    sess_options.inter_op_num_threads = sess_conf.get("inter_op_num_threads", 0)

    sess = ort.InferenceSession(model_path,
                                providers=providers,
                                sess_options=sess_options)
    return sess
