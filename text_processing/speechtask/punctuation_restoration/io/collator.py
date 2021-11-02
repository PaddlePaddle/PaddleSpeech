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
import numpy as np

__all__ = ["TextCollator"]


class TextCollator():
    def __init__(self, padding_value):
        self.padding_value = padding_value

    def __call__(self, batch):
        """batch examples
        Args:
            batch ([List]): batch is (text, punctuation)
                text (List[int] ) shape (batch, L)
                punctuation (List[int] or str): shape (batch, L)
        Returns:
            tuple(text, punctuation): batched data.
                text : (B, Lmax)
                punctuation : (B, Lmax)
        """
        texts = []
        punctuations = []
        for text, punctuation in batch:

            texts.append(text)
            punctuations.append(punctuation)

        #[B, T, D]
        x_pad = self.pad_sequence(texts).astype(np.int64)
        # print(x_pad.shape)
        # pad_list(audios, 0.0).astype(np.float32)
        # ilens = np.array(audio_lens).astype(np.int64)
        y_pad = self.pad_sequence(punctuations).astype(np.int64)
        # print(y_pad.shape)
        # olens = np.array(text_lens).astype(np.int64)
        return x_pad, y_pad

    def pad_sequence(self, sequences):
        # assuming trailing dimensions and type of all the Tensors
        # in sequences are same and fetching those from sequences[0]
        max_len = max([len(s) for s in sequences])
        out_dims = (len(sequences), max_len)

        out_tensor = np.full(out_dims,
                             self.padding_value)  #, dtype=sequences[0].dtype)
        for i, tensor in enumerate(sequences):
            length = len(tensor)
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :length] = tensor

        return out_tensor
