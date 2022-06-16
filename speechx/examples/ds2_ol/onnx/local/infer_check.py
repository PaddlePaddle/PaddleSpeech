#!/usr/bin/env python3
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import os
import pickle

import numpy as np
import onnxruntime
import paddle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input_file',
        type=str,
        default="static_ds2online_inputs.pickle",
        help="aishell ds2 input data file. For wenetspeech, we only feed for infer model", )
    parser.add_argument(
        '--model_type',
        type=str,
        default="aishell",
        help="aishell(1024) or wenetspeech(2048)", )
    parser.add_argument(
        '--model_dir', type=str, default=".", help="paddle model dir.")
    parser.add_argument(
        '--model_prefix',
        type=str,
        default="avg_1.jit",
        help="paddle model prefix.")
    parser.add_argument(
        '--onnx_model',
        type=str,
        default='./model.old.onnx',
        help="onnx model.")

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = parse_args()

    # input and output
    with open(FLAGS.input_file, 'rb') as f:
        iodict = pickle.load(f)
        print(iodict.keys())


    audio_chunk = iodict['audio_chunk']
    audio_chunk_lens = iodict['audio_chunk_lens']
    chunk_state_h_box = iodict['chunk_state_h_box']
    chunk_state_c_box = iodict['chunk_state_c_bos']
    print("raw state shape: ", chunk_state_c_box.shape)

    if FLAGS.model_type == 'wenetspeech':
        chunk_state_h_box = np.repeat(chunk_state_h_box, 2, axis=-1)
        chunk_state_c_box = np.repeat(chunk_state_c_box, 2, axis=-1)
    print("state shape: ", chunk_state_c_box.shape)

    # paddle
    model = paddle.jit.load(os.path.join(FLAGS.model_dir, FLAGS.model_prefix))
    res_chunk, res_lens, chunk_state_h, chunk_state_c = model(
        paddle.to_tensor(audio_chunk),
        paddle.to_tensor(audio_chunk_lens),
        paddle.to_tensor(chunk_state_h_box),
        paddle.to_tensor(chunk_state_c_box), )

    # onnxruntime
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    sess = onnxruntime.InferenceSession(FLAGS.onnx_model, sess_options=options)
    ort_res_chunk, ort_res_lens, ort_chunk_state_h, ort_chunk_state_c = sess.run(
        ['softmax_0.tmp_0', 'tmp_5', 'concat_0.tmp_0', 'concat_1.tmp_0'], {
            "audio_chunk": audio_chunk,
            "audio_chunk_lens": audio_chunk_lens,
            "chunk_state_h_box": chunk_state_h_box,
            "chunk_state_c_box": chunk_state_c_box
        })

    print(sess.end_profiling())

    # assert paddle equal ort
    print(np.allclose(ort_res_chunk, res_chunk, atol=1e-6))
    print(np.allclose(ort_res_lens, res_lens, atol=1e-6))

    if FLAGS.model_type == 'aishell':
        print(np.allclose(ort_chunk_state_h, chunk_state_h, atol=1e-6))
        print(np.allclose(ort_chunk_state_c, chunk_state_c, atol=1e-6))
