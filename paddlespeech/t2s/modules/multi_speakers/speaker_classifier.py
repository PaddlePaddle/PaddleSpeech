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
# Modified from Cross-Lingual-Voice-Cloning(https://github.com/deterministic-algorithms-lab/Cross-Lingual-Voice-Cloning)

from paddle import nn
import paddle
from typeguard import check_argument_types

class SpeakerClassifier(nn.Layer):
    
    def __init__(self, idim: int, hidden_sc_dim: int, spk_num: int, ):
        assert check_argument_types()
        super().__init__()
        # store hyperparameters
        self.idim = idim
        self.hidden_sc_dim = hidden_sc_dim
        self.spk_num = spk_num

        self.model = nn.Sequential(nn.Linear(self.idim, self.hidden_sc_dim),
                                   nn.Linear(self.hidden_sc_dim, self.spk_num))
    
    def parse_outputs(self, out, text_lengths):
        mask = paddle.arange(out.shape[1]).expand([out.shape[0], out.shape[1]]) < text_lengths.unsqueeze(1)
        out = paddle.transpose(out, perm=[2, 0, 1])
        out = out * mask
        out = paddle.transpose(out, perm=[1, 2, 0])
        return out

    def forward(self, encoder_outputs, text_lengths):
        """
        encoder_outputs = [batch_size, seq_len, encoder_embedding_size]
        text_lengths = [batch_size]
        
        log probabilities of speaker classification = [batch_size, seq_len, spk_num]
        """
        
        out = self.model(encoder_outputs) 
        out = self.parse_outputs(out, text_lengths)
        return out
