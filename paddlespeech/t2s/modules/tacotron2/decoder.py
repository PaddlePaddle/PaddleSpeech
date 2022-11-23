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
"""Tacotron2 decoder related modules."""
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.modules.tacotron2.attentions import AttForwardTA


class Prenet(nn.Layer):
    """Prenet module for decoder of Spectrogram prediction network.

    This is a module of Prenet in the decoder of Spectrogram prediction network,
    which described in `Natural TTS
    Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Prenet preforms nonlinear conversion
    of inputs before input to auto-regressive lstm,
    which helps to learn diagonal attentions.

    Notes
    ----------
    This module alway applies dropout even in evaluation.
    See the detail in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, idim, n_layers=2, n_units=256, dropout_rate=0.5):
        """Initialize prenet module.

        Args:
            idim (int): 
                Dimension of the inputs.
            odim (int): 
                Dimension of the outputs.
            n_layers (int, optional): 
                The number of prenet layers.
            n_units (int, optional): 
                The number of prenet units.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.prenet = nn.LayerList()
        for layer in range(n_layers):
            n_inputs = idim if layer == 0 else n_units
            self.prenet.append(
                nn.Sequential(nn.Linear(n_inputs, n_units), nn.ReLU()))

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): 
                Batch of input tensors (B, ..., idim).

        Returns: 
            Tensor: Batch of output tensors (B, ..., odim).

        """
        for i in range(len(self.prenet)):
            # F.dropout 引入了随机, tacotron2 的 dropout 是不能去掉的
            x = F.dropout(self.prenet[i](x))
        return x


class Postnet(nn.Layer):
    """Postnet module for Spectrogram prediction network.

    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail sturcture of spectrogram.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
            self,
            idim,
            odim,
            n_layers=5,
            n_chans=512,
            n_filts=5,
            dropout_rate=0.5,
            use_batch_norm=True, ):
        """Initialize postnet module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..
        """
        super().__init__()
        self.postnet = nn.LayerList()
        for layer in range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet.append(
                    nn.Sequential(
                        nn.Conv1D(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias_attr=False, ),
                        nn.BatchNorm1D(ochans),
                        nn.Tanh(),
                        nn.Dropout(dropout_rate), ))
            else:
                self.postnet.append(
                    nn.Sequential(
                        nn.Conv1D(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias_attr=False, ),
                        nn.Tanh(),
                        nn.Dropout(dropout_rate), ))
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet.append(
                nn.Sequential(
                    nn.Conv1D(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias_attr=False, ),
                    nn.BatchNorm1D(odim),
                    nn.Dropout(dropout_rate), ))
        else:
            self.postnet.append(
                nn.Sequential(
                    nn.Conv1D(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias_attr=False, ),
                    nn.Dropout(dropout_rate), ))

    def forward(self, xs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).
        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).
        """
        for i in range(len(self.postnet)):
            xs = self.postnet[i](xs)
        return xs


class ZoneOutCell(nn.Layer):
    """ZoneOut Cell module.
    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.
    Examples
    ----------
        >>> lstm = paddle.nn.LSTMCell(16, 32)
        >>> lstm = ZoneOutCell(lstm, 0.5)
    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305
    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch
    """

    def __init__(self, cell, zoneout_rate=0.1):
        """Initialize zone out cell module.

        Args:
            cell (nn.Layer): Paddle recurrent cell module
                e.g. `paddle.nn.LSTMCell`.
            zoneout_rate (float, optional): Probability of zoneout from 0.0 to 1.0.
        """
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_rate = zoneout_rate
        if zoneout_rate > 1.0 or zoneout_rate < 0.0:
            raise ValueError(
                "zoneout probability must be in the range from 0.0 to 1.0.")

    def forward(self, inputs, hidden):
        """Calculate forward propagation.

        Args:
            inputs (Tensor): 
                Batch of input tensor (B, input_size).
            hidden (tuple):
                - Tensor: Batch of initial hidden states (B, hidden_size).
                - Tensor: Batch of initial cell states (B, hidden_size).
        Returns:
            Tensor:
                Batch of next hidden states (B, hidden_size).
            tuple:
                - Tensor: Batch of next hidden states (B, hidden_size).
                - Tensor: Batch of next cell states (B, hidden_size).
        """
        # we only use the second output of LSTMCell in paddle
        _, next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
        # to have the same output format with LSTMCell in paddle
        return next_hidden[0], next_hidden

    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])
        if self.training:
            mask = paddle.bernoulli(paddle.ones([*paddle.shape(h)]) * prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class Decoder(nn.Layer):
    """Decoder module of Spectrogram prediction network.
    This is a module of decoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS
    Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The decoder generates the sequence of
    features from the sequence of the hidden states.
    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(
            self,
            idim,
            odim,
            att,
            dlayers=2,
            dunits=1024,
            prenet_layers=2,
            prenet_units=256,
            postnet_layers=5,
            postnet_chans=512,
            postnet_filts=5,
            output_activation_fn=None,
            cumulate_att_w=True,
            use_batch_norm=True,
            use_concate=True,
            dropout_rate=0.5,
            zoneout_rate=0.1,
            reduction_factor=1, ):
        """Initialize Tacotron2 decoder module.

        Args:
            idim (int): 
                Dimension of the inputs.
            odim (int): 
                Dimension of the outputs.
            att (nn.Layer): 
                Instance of attention class.
            dlayers (int, optional): 
                The number of decoder lstm layers.
            dunits (int, optional): 
                The number of decoder lstm units.
            prenet_layers (int, optional): 
                The number of prenet layers.
            prenet_units (int, optional): 
                The number of prenet units.
            postnet_layers (int, optional): 
                The number of postnet layers.
            postnet_filts (int, optional): 
                The number of postnet filter size.
            postnet_chans (int, optional): 
                The number of postnet filter channels.
            output_activation_fn (nn.Layer, optional): 
                Activation function for outputs.
            cumulate_att_w (bool, optional): 
                Whether to cumulate previous attention weight.
            use_batch_norm (bool, optional): 
                Whether to use batch normalization.
            use_concate (bool, optional):
                Whether to concatenate encoder embedding with decoder lstm outputs.
            dropout_rate (float, optional):
                Dropout rate.
            zoneout_rate (float, optional):
                Zoneout rate.
            reduction_factor (int, optional):
                Reduction factor.
        """
        super().__init__()

        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.output_activation_fn = output_activation_fn
        self.cumulate_att_w = cumulate_att_w
        self.use_concate = use_concate
        self.reduction_factor = reduction_factor

        # check attention type
        if isinstance(self.att, AttForwardTA):
            self.use_att_extra_inputs = True
        else:
            self.use_att_extra_inputs = False

        # define lstm network
        prenet_units = prenet_units if prenet_layers != 0 else odim
        self.lstm = nn.LayerList()
        for layer in range(dlayers):
            iunits = idim + prenet_units if layer == 0 else dunits
            lstm = nn.LSTMCell(iunits, dunits)
            if zoneout_rate > 0.0:
                lstm = ZoneOutCell(lstm, zoneout_rate)
            self.lstm.append(lstm)

        # define prenet
        if prenet_layers > 0:
            self.prenet = Prenet(
                idim=odim,
                n_layers=prenet_layers,
                n_units=prenet_units,
                dropout_rate=dropout_rate, )
        else:
            self.prenet = None

        # define postnet
        if postnet_layers > 0:
            self.postnet = Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate, )
        else:
            self.postnet = None

        # define projection layers
        iunits = idim + dunits if use_concate else dunits
        self.feat_out = nn.Linear(
            iunits, odim * reduction_factor, bias_attr=False)
        self.prob_out = nn.Linear(iunits, reduction_factor)

    def _zero_state(self, hs):
        init_hs = paddle.zeros([paddle.shape(hs)[0], self.lstm[0].hidden_size])
        return init_hs

    def forward(self, hs, hlens, ys):
        """Calculate forward propagation.

        Args:
            hs (Tensor): 
                Batch of the sequences of padded hidden states (B, Tmax, idim).
            hlens (Tensor(int64) padded): 
                Batch of lengths of each input batch (B,).
            ys (Tensor): 
                Batch of the sequences of padded target features (B, Lmax, odim).

        Returns:
            Tensor: 
                Batch of output tensors after postnet (B, Lmax, odim).
            Tensor: 
                Batch of output tensors before postnet (B, Lmax, odim).
            Tensor: 
                Batch of logits of stop prediction (B, Lmax).
            Tensor: 
                Batch of attention weights (B, Lmax, Tmax).
            
        Note: 
            This computation is performed in teacher-forcing manner.
        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1::self.reduction_factor]

        # length list should be list of int
        # hlens = list(map(int, hlens))

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in range(1, len(self.lstm)):
            c_list.append(self._zero_state(hs))
            z_list.append(self._zero_state(hs))
        prev_out = paddle.zeros([paddle.shape(hs)[0], self.odim])

        # initialize attention
        prev_att_ws = []
        prev_att_w = paddle.zeros(paddle.shape(hlens))
        prev_att_ws.append(prev_att_w)
        self.att.reset()

        # loop for an output sequence
        outs, logits, att_ws = [], [], []
        for y in ys.transpose([1, 0, 2]):
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_ws[-1],
                                        prev_out)
            else:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_ws[-1])
            prenet_out = self.prenet(
                prev_out) if self.prenet is not None else prev_out
            xs = paddle.concat([att_c, prenet_out], axis=1)
            # we only use the second output of LSTMCell in paddle
            _, next_hidden = self.lstm[0](xs, (z_list[0], c_list[0]))
            z_list[0], c_list[0] = next_hidden
            for i in range(1, len(self.lstm)):
                # we only use the second output of LSTMCell in paddle
                _, next_hidden = self.lstm[i](z_list[i - 1],
                                              (z_list[i], c_list[i]))
                z_list[i], c_list[i] = next_hidden
            zcs = (paddle.concat([z_list[-1], att_c], axis=1)
                   if self.use_concate else z_list[-1])
            outs.append(
                self.feat_out(zcs).reshape([paddle.shape(hs)[0], self.odim, -1
                                            ]))
            logits.append(self.prob_out(zcs))
            att_ws.append(att_w)
            # teacher forcing
            prev_out = y
            if self.cumulate_att_w and paddle.sum(prev_att_w) != 0:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
            prev_att_ws.append(prev_att_w)
        # (B, Lmax)
        logits = paddle.concat(logits, axis=1)
        # (B, odim, Lmax) 
        before_outs = paddle.concat(outs, axis=2)
        # (B, Lmax, Tmax)
        att_ws = paddle.stack(att_ws, axis=1)

        if self.reduction_factor > 1:
            # (B, odim, Lmax)
            before_outs = before_outs.reshape(
                [paddle.shape(before_outs)[0], self.odim, -1])

        if self.postnet is not None:
            # (B, odim, Lmax)
            after_outs = before_outs + self.postnet(before_outs)
        else:
            after_outs = before_outs
        # (B, Lmax, odim)
        before_outs = before_outs.transpose([0, 2, 1])
        # (B, Lmax, odim)
        after_outs = after_outs.transpose([0, 2, 1])
        logits = logits

        # apply activation function for scaling
        if self.output_activation_fn is not None:
            before_outs = self.output_activation_fn(before_outs)
            after_outs = self.output_activation_fn(after_outs)

        return after_outs, before_outs, logits, att_ws

    def inference(
            self,
            h,
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=None,
            forward_window=None, ):
        """Generate the sequence of features given the sequences of characters.
        Args:
            h(Tensor): 
                Input sequence of encoder hidden states (T, C).
            threshold(float, optional, optional): 
                Threshold to stop generation. (Default value = 0.5)
            minlenratio(float, optional, optional): 
                Minimum length ratio. If set to 1.0 and the length of input is 10,
                the minimum length of outputs will be 10 * 1 = 10. (Default value = 0.0)
            maxlenratio(float, optional, optional):
                 Minimum length ratio. If set to 10 and the length of input is 10,
                the maximum length of outputs will be 10 * 10 = 100. (Default value = 0.0)
            use_att_constraint(bool, optional): 
                Whether to apply attention constraint introduced in `Deep Voice 3`_. (Default value = False)
            backward_window(int, optional): 
                Backward window size in attention constraint. (Default value = None)
            forward_window(int, optional):  
                    (Default value = None)

        Returns:
            Tensor: 
                Output sequence of features (L, odim).
            Tensor: 
                Output sequence of stop probabilities (L,).
            Tensor: 
                Attention weights (L, T).

        Note: 
            This computation is performed in auto-regressive manner.
    .. _`Deep Voice 3`: https://arxiv.org/abs/1710.07654
        """
        # setup

        assert len(paddle.shape(h)) == 2
        hs = h.unsqueeze(0)
        ilens = paddle.shape(h)[0]
        # 本来 maxlen 和 minlen 外面有 int()，防止动转静的问题此处删除
        maxlen = paddle.shape(h)[0] * maxlenratio
        minlen = paddle.shape(h)[0] * minlenratio
        # 本来是直接使用 threshold 的，此处为了防止动转静的问题把 threshold 转成 tensor
        threshold = paddle.ones([1]) * threshold

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in range(1, len(self.lstm)):
            c_list.append(self._zero_state(hs))
            z_list.append(self._zero_state(hs))
        prev_out = paddle.zeros([1, self.odim])

        # initialize attention
        prev_att_ws = []
        prev_att_w = paddle.zeros([ilens])
        prev_att_ws.append(prev_att_w)

        self.att.reset()

        # setup for attention constraint
        if use_att_constraint:
            last_attended_idx = 0
        else:
            last_attended_idx = None

        # loop for an output sequence
        idx = 0
        outs, att_ws, probs = [], [], []
        prob = paddle.zeros([1])
        while True:
            # updated index
            idx += self.reduction_factor

            # decoder calculation
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(
                    hs,
                    ilens,
                    z_list[0],
                    prev_att_ws[-1],
                    prev_out,
                    last_attended_idx=last_attended_idx,
                    backward_window=backward_window,
                    forward_window=forward_window, )
            else:
                att_c, att_w = self.att(
                    hs,
                    ilens,
                    z_list[0],
                    prev_att_ws[-1],
                    last_attended_idx=last_attended_idx,
                    backward_window=backward_window,
                    forward_window=forward_window, )

            att_ws.append(att_w)
            prenet_out = self.prenet(
                prev_out) if self.prenet is not None else prev_out
            xs = paddle.concat([att_c, prenet_out], axis=1)
            # we only use the second output of LSTMCell in paddle
            _, next_hidden = self.lstm[0](xs, (z_list[0], c_list[0]))

            z_list[0], c_list[0] = next_hidden
            for i in range(1, len(self.lstm)):
                # we only use the second output of LSTMCell in paddle
                _, next_hidden = self.lstm[i](z_list[i - 1],
                                              (z_list[i], c_list[i]))
                z_list[i], c_list[i] = next_hidden
            zcs = (paddle.concat([z_list[-1], att_c], axis=1)
                   if self.use_concate else z_list[-1])
            # [(1, odim, r), ...]
            outs.append(self.feat_out(zcs).reshape([1, self.odim, -1]))

            prob = F.sigmoid(self.prob_out(zcs))[0]
            probs.append(prob)

            if self.output_activation_fn is not None:
                prev_out = self.output_activation_fn(
                    outs[-1][:, :, -1])  # (1, odim)
            else:
                prev_out = outs[-1][:, :, -1]  # (1, odim)
            if self.cumulate_att_w and paddle.sum(prev_att_w) != 0:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
            prev_att_ws.append(prev_att_w)
            if use_att_constraint:
                last_attended_idx = int(att_w.argmax())

            # tacotron2 ljspeech 动转静的问题应该是这里没有正确判断 prob >= threshold 导致的
            if prob >= threshold or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                break
            """
            仅解开 665~667 行的代码块，动转静时会卡死，但是动态图时可以正确生成音频，证明模型没问题
            同时解开 665~667 行 和 668 ~ 670 行的代码块，动转静时不会卡死，但是生成的音频末尾有多余的噪声
            证明动转静没有进入 prob >= threshold 的判断，但是静态图可以进入 prob >= threshold 并退出循环
            动转静时是通过 idx >= maxlen 退出循环（所以没有这个逻辑的时候会一直循环，也就是卡死），
            没有在模型判断该结束的时候结束，而是在超出最大长度时结束，所以合成的音频末尾有很长的额外预测的噪声
            动转静用 prob <= threshold 的条件可以退出循环（虽然结果不正确），证明条件参数的类型本身没问题，可能是 prob 有问题
            """
            # if prob >= threshold:
            #     print("prob >= threshold")
            #     break
            # elif idx >= maxlen:
            #     print("idx >= maxlen")
            #     break

        # (1, odim, L)
        outs = paddle.concat(outs, axis=2)
        if self.postnet is not None:
            # (1, odim, L)
            outs = outs + self.postnet(outs)
        # (L, odim)
        outs = outs.transpose([0, 2, 1]).squeeze(0)
        probs = paddle.concat(probs, axis=0)
        att_ws = paddle.concat(att_ws, axis=0)

        if self.output_activation_fn is not None:
            outs = self.output_activation_fn(outs)

        return outs, probs, att_ws

    def calculate_all_attentions(self, hs, hlens, ys):
        """Calculate all of the attention weights.

        Args:
            hs (Tensor): 
                Batch of the sequences of padded hidden states (B, Tmax, idim).
            hlens (Tensor(int64)): 
                Batch of lengths of each input batch (B,).
            ys (Tensor): 
                Batch of the sequences of padded target features (B, Lmax, odim).

        Returns:
            numpy.ndarray:
                Batch of attention weights (B, Lmax, Tmax).
    
        Note:
            This computation is performed in teacher-forcing manner.
        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1::self.reduction_factor]

        # length list should be list of int
        hlens = list(map(int, hlens))

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in range(1, len(self.lstm)):
            c_list.append(self._zero_state(hs))
            z_list.append(self._zero_state(hs))
        prev_out = paddle.zeros([paddle.shape(hs)[0], self.odim])

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        att_ws = []
        for y in ys.transpose([1, 0, 2]):
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w,
                                        prev_out)
            else:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            att_ws.append(att_w)
            prenet_out = self.prenet(
                prev_out) if self.prenet is not None else prev_out
            xs = paddle.concat([att_c, prenet_out], axis=1)
            # we only use the second output of LSTMCell in paddle
            _, next_hidden = self.lstm[0](xs, (z_list[0], c_list[0]))
            z_list[0], c_list[0] = next_hidden
            for i in range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](z_list[i - 1],
                                                    (z_list[i], c_list[i]))
            # teacher forcing
            prev_out = y
            if self.cumulate_att_w and prev_att_w is not None:
                # Note: error when use +=
                prev_att_w = prev_att_w + att_w
            else:
                prev_att_w = att_w
        # (B, Lmax, Tmax)
        att_ws = paddle.stack(att_ws, axis=1)

        return att_ws
