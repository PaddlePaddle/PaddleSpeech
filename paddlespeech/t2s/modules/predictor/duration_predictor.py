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
# Modified from espnet(https://github.com/espnet/espnet)
"""Duration predictor related modules."""
import paddle
from paddle import nn

from paddlespeech.t2s.modules.layer_norm import LayerNorm
from paddlespeech.t2s.modules.masked_fill import masked_fill


class DurationPredictor(nn.Layer):
    """Duration predictor module.

    This is a module of duration predictor described
    in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note
    ----------
    The calculation domain of outputs is different
    between in `forward` and in `inference`. In `forward`,
    the outputs are calculated in log domain but in `inference`,
    those are calculated in linear domain.

    """

    def __init__(self,
                 idim,
                 n_layers=2,
                 n_chans=384,
                 kernel_size=3,
                 dropout_rate=0.1,
                 offset=1.0):
        """Initilize duration predictor module.

        Parameters
        ----------
        idim : int
            Input dimension.
        n_layers : int, optional
                Number of convolutional layers.
        n_chans : int, optional
            Number of channels of convolutional layers.
        kernel_size : int, optional
            Kernel size of convolutional layers.
        dropout_rate : float, optional
                Dropout rate.
        offset : float, optional
            Offset value to avoid nan in log domain.

        """
        super().__init__()
        self.offset = offset
        self.conv = nn.LayerList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv.append(
                nn.Sequential(
                    nn.Conv1D(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2, ),
                    nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    nn.Dropout(dropout_rate), ))
        self.linear = nn.Linear(n_chans, 1, bias_attr=True)

    def _forward(self, xs, x_masks=None, is_inference=False):
        # (B, idim, Tmax)
        xs = xs.transpose([0, 2, 1])
        # (B, C, Tmax)
        for f in self.conv:
            xs = f(xs)

        # NOTE: calculate in log domain
        # (B, Tmax)
        xs = self.linear(xs.transpose([0, 2, 1])).squeeze(-1)

        if is_inference:
            # NOTE: calculate in linear domain
            xs = paddle.clip(paddle.round(xs.exp() - self.offset), min=0)

        if x_masks is not None:
            xs = masked_fill(xs, x_masks, 0.0)

        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.

        Parameters
        ----------
        xs : Tensor
            Batch of input sequences (B, Tmax, idim).
        x_masks : ByteTensor, optional
            Batch of masks indicating padded part (B, Tmax).

        Returns
        ----------
            Tensor
                Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.

        Parameters
        ----------
        xs : Tensor
            Batch of input sequences (B, Tmax, idim).
        x_masks : Tensor(bool), optional
            Batch of masks indicating padded part (B, Tmax).

        Returns
        ----------
        Tensor
            Batch of predicted durations in linear domain int64 (B, Tmax).
        """
        return self._forward(xs, x_masks, True)


class DurationPredictorLoss(nn.Layer):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction="mean"):
        """Initilize duration predictor loss module.

        Parameters
        ----------
        offset : float, optional
            Offset value to avoid nan in log domain.
        reduction : str
            Reduction type in loss calculation.
        """
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Parameters
        ----------
        outputs : Tensor
            Batch of prediction durations in log domain (B, T)
        targets : Tensor
            Batch of groundtruth durations in linear domain (B, T)

        Returns
        ----------
        Tensor
            Mean squared error loss value.

        Note
        ----------
        `outputs` is in log domain but `targets` is in linear domain.
        """
        # NOTE: outputs is in log domain while targets in linear
        targets = paddle.log(targets.cast(dtype='float32') + self.offset)
        loss = self.criterion(outputs, targets)

        return loss
