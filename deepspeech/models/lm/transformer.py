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
from typing import Any
from typing import List
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from deepspeech.modules.encoder import TransformerEncoder


class TransformerLM(nn.Layer):
    def __init__(
            self,
            vocab_size: int,
            pos_enc: str=None,
            embed_unit: int=128,
            att_unit: int=256,
            head: int=2,
            unit: int=1024,
            layer: int=4,
            dropout_rate: float=0.5, ):
        super().__init__()
        if pos_enc == "sinusoidal":
            pos_enc_layer_type = "abs_pos"
        elif pos_enc is None:
            #TODO
            raise ValueError(f"unknown pos-enc option: {pos_enc}")
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")

        self.embed = nn.Embedding(vocab_size, embed_unit)
        self.encoder = TransformerEncoder(
            input_size=embed_unit,
            output_size=att_unit,
            attention_heads=head,
            linear_units=unit,
            num_blocks=layer,
            dropout_rate=dropout_rate,
            input_layer="linear",
            pos_enc_layer_type=pos_enc_layer_type,
            concat_after=False,
            static_chunk_size=1,
            use_dynamic_chunk=False,
            use_dynamic_left_chunk=False, )

        self.decoder = nn.Linear(att_unit, vocab_size)

        self.load_parameters()

    def load_parameters(self):
        model_dict = paddle.load("transformerLM.pdparams")
        self.set_state_dict(model_dict)

    def forward(self, input: paddle.Tensor,
                hidden: None) -> Tuple[paddle.Tensor, None]:

        x = self.embed(input)
        x_len = self._target_len(input)
        h, _ = self.encoder(x, x_len)
        y = self.decoder(h)
        return y, None

    def score(
            self,
            y: paddle.Tensor,
            subsampling_cache,
            state: Any,
            offset: int, ) -> Tuple[paddle.Tensor, Any]:
        # y, the chunk input
        y = y.unsqueeze(0)
        subsampling_cache = subsampling_cache
        conformer_cnn_cache = None
        elayers_output_cache = state
        required_cache_size = -1
        y = self.embed(y)
        h, r_subsampling_cache, r_elayers_output_cache, r_conformer_cnn_cache = self.encoder.forward_chunk(
            y, offset, required_cache_size, subsampling_cache,
            elayers_output_cache, conformer_cnn_cache)
        h = self.decoder(h[:, -1])
        logp = F.log_softmax(h).squeeze(0)
        return h, r_subsampling_cache, r_elayers_output_cache

    def batch_score(
            self,
            ys: paddle.Tensor,
            subsampling_caches: List[Any],
            encoder_states: List[Any],
            offset: int, ) -> Tuple[paddle.Tensor, List[Any]]:
        #ys, the batch chunk input
        n_batch = ys.shape[0]
        n_layers = len(self.encoder.encoders)
        hs = []
        new_subsampling_states = []
        new_encoder_states = []
        for i in range(n_batch):
            y = ys[i:i + 1, :]
            subsampling_cache = subsampling_caches[i]
            elayers_output_cache = encoder_states[i]
            conformer_cnn_cache = None
            required_cache_size = -1
            y = self.embed(y)
            h, r_subsampling_cache, r_elayers_output_cache, r_conformer_cnn_cache = self.encoder.forward_chunk(
                y, offset, required_cache_size, subsampling_cache,
                elayers_output_cache, conformer_cnn_cache)
            h = self.decoder(h[:, -1])
            hs.append(h)
            new_subsampling_states.append(r_subsampling_cache)
            new_encoder_states.append(r_elayers_output_cache)
        hs = paddle.concat(hs, axis=0)
        hs = F.log_softmax(hs)
        return hs, new_subsampling_states, new_encoder_states


if __name__ == "__main__":
    tlm = TransformerLM(
        vocab_size=5002,
        pos_enc='sinusoidal',
        embed_unit=128,
        att_unit=512,
        head=8,
        unit=2048,
        layer=16,
        dropout_rate=0.5, )
    paddle.set_device("cpu")

    tlm.eval()
    """
    #Test the score
    input2 = np.array([5])
    input2 = paddle.to_tensor(input2)
    output, sub_cache, cache =tlm.score(input2, None, None, 0)

    input3 = np.array([10])
    input3 = paddle.to_tensor(input3)
    output, sub_cache, cache = tlm.score(input3, sub_cache, cache, 1)

    input4 = np.array([7])
    input4 = paddle.to_tensor(input4)
    output, sub_cache, cache = tlm.score(input4, sub_cache, cache, 2)
    print ("output", output)
    """
    #Test the batch score
    batch_size = 2
    offset = 0
    inp2 = np.array([[5], [10]])
    inp2 = paddle.to_tensor(inp2)
    output, subsampling_caches, encoder_caches = tlm.batch_score(
        inp2, [None] * batch_size, [None] * batch_size, offset)

    offset += 1
    inp3 = np.array([[100], [30]])
    inp3 = paddle.to_tensor(inp3)
    output, subsampling_caches, encoder_caches = tlm.batch_score(
        inp3, subsampling_caches, encoder_caches, offset)
    print("output", output)
    #print("cache", cache)
    #np.save("output_pd.npy", output)
