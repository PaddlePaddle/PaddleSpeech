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
"""Attention modules for RNN."""
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.modules.masked_fill import masked_fill
from paddlespeech.t2s.modules.nets_utils import make_pad_mask


def _apply_attention_constraint(e,
                                last_attended_idx,
                                backward_window=1,
                                forward_window=3):
    """Apply monotonic attention constraint.

    This function apply the monotonic attention constraint
    introduced in `Deep Voice 3: Scaling
    Text-to-Speech with Convolutional Sequence Learning`_.

    Args:
        e(Tensor): 
            Attention energy before applying softmax (1, T).
        last_attended_idx(int): 
            The index of the inputs of the last attended [0, T].
        backward_window(int, optional, optional): 
            Backward window size in attention constraint. (Default value = 1)
        forward_window(int, optional, optional): 
            Forward window size in attetion constraint. (Default value = 3)

    Returns:
        Tensor: Monotonic constrained attention energy (1, T).

    .. _`Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning`:
        https://arxiv.org/abs/1710.07654

    """
    # for dygraph to static graph
    # if e.shape[0] != 1:
    #     raise NotImplementedError(
    #         "Batch attention constraining is not yet supported.")
    backward_idx = paddle.cast(
        last_attended_idx - backward_window, dtype='int64')
    forward_idx = paddle.cast(last_attended_idx + forward_window, dtype='int64')
    if backward_idx > 0:
        e[:, :backward_idx] = -float("inf")
    if forward_idx < paddle.shape(e)[1]:
        e[:, forward_idx:] = -float("inf")
    return e


class AttLoc(nn.Layer):
    """location-aware attention module.

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    Args:
        eprojs (int): 
            projection-units of encoder
        dunits (int): 
            units of decoder
        att_dim (int): 
            attention dimension
        aconv_chans (int): 
            channels of attention convolution
        aconv_filts (int): 
            filter size of attention convolution
        han_mode (bool): 
            flag to swith on mode of hierarchical attention and not store pre_compute_enc_h
    """

    def __init__(self,
                 eprojs,
                 dunits,
                 att_dim,
                 aconv_chans,
                 aconv_filts,
                 han_mode=False):
        super().__init__()
        self.mlp_enc = nn.Linear(eprojs, att_dim)
        self.mlp_dec = nn.Linear(dunits, att_dim, bias_attr=False)
        self.mlp_att = nn.Linear(aconv_chans, att_dim, bias_attr=False)
        self.loc_conv = nn.Conv2D(
            1,
            aconv_chans,
            (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias_attr=False, )
        self.gvec = nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(
            self,
            enc_hs_pad,
            enc_hs_len,
            dec_z,
            att_prev,
            scaling=2.0,
            last_attended_idx=-1,
            backward_window=1,
            forward_window=3, ):
        """Calculate AttLoc forward propagation.
        Args:
            enc_hs_pad(Tensor): 
                padded encoder hidden state (B, T_max, D_enc)
            enc_hs_len(Tensor): 
                padded encoder hidden state length (B)
            dec_z(Tensor dec_z): 
                decoder hidden state (B, D_dec)
            att_prev(Tensor): 
                previous attention weight (B, T_max)
            scaling(float, optional): 
                scaling parameter before applying softmax (Default value = 2.0)
            forward_window(Tensor, optional): 
                    forward window size when constraining attention (Default value = 3)
            last_attended_idx(int, optional): 
                index of the inputs of the last attended (Default value = None)
            backward_window(int, optional): 
                backward window size in attention constraint (Default value = 1)
            forward_window(int, optional): 
                    forward window size in attetion constraint (Default value = 3)
        Returns:
            Tensor: 
                attention weighted encoder state (B, D_enc)
            Tensor: 
                previous attention weights (B, T_max)
        """
        batch = paddle.shape(enc_hs_pad)[0]
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            # (utt, frame, hdim)
            self.enc_h = enc_hs_pad
            self.h_length = paddle.shape(self.enc_h)[1]
            # (utt, frame, att_dim)
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = paddle.zeros([batch, self.dunits])
        else:
            dec_z = dec_z.reshape([batch, self.dunits])

        # initialize attention weight with uniform dist.
        if paddle.sum(att_prev) == 0:
            # if no bias, 0 0-pad goes 0
            att_prev = 1.0 - make_pad_mask(enc_hs_len)
            att_prev = att_prev / enc_hs_len.unsqueeze(-1).astype(
                att_prev.dtype)

        # att_prev: (utt, frame) -> (utt, 1, 1, frame)
        # -> (utt, att_conv_chans, 1, frame)
        att_conv = self.loc_conv(att_prev.reshape([batch, 1, 1, self.h_length]))
        # att_conv: (utt, att_conv_chans, 1, frame) -> (utt, frame, att_conv_chans)
        att_conv = att_conv.squeeze(2).transpose([0, 2, 1])
        # att_conv: (utt, frame, att_conv_chans) -> (utt, frame, att_dim)
        att_conv = self.mlp_att(att_conv)
        # dec_z_tiled: (utt, frame, att_dim)        
        dec_z_tiled = self.mlp_dec(dec_z).reshape([batch, 1, self.att_dim])

        # dot with gvec
        # (utt, frame, att_dim) -> (utt, frame)
        e = paddle.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)
        e = self.gvec(e).squeeze(2)

        # NOTE: consider zero padding when compute w.
        if self.mask is None:
            self.mask = make_pad_mask(enc_hs_len)

        e = masked_fill(e, self.mask, -float("inf"))
        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx != -1:
            e = _apply_attention_constraint(e, last_attended_idx,
                                            backward_window, forward_window)

        w = F.softmax(scaling * e, axis=1)

        # weighted sum over frames
        # utt x hdim
        c = paddle.sum(
            self.enc_h * w.reshape([batch, self.h_length, 1]), axis=1)
        return c, w


class AttForward(nn.Layer):
    """Forward attention module.
    Reference
    ----------
    Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    Args:
        eprojs (int): 
            projection-units of encoder
        dunits (int): 
            units of decoder
        att_dim (int): 
            attention dimension
        aconv_chans (int): 
            channels of attention convolution
        aconv_filts (int): 
            filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super().__init__()
        self.mlp_enc = nn.Linear(eprojs, att_dim)
        self.mlp_dec = nn.Linear(dunits, att_dim, bias_attr=False)
        self.mlp_att = nn.Linear(aconv_chans, att_dim, bias_attr=False)
        self.loc_conv = nn.Conv2D(
            1,
            aconv_chans,
            (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias_attr=False, )
        self.gvec = nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(
            self,
            enc_hs_pad,
            enc_hs_len,
            dec_z,
            att_prev,
            scaling=1.0,
            last_attended_idx=None,
            backward_window=1,
            forward_window=3, ):
        """Calculate AttForward forward propagation.

        Args:
            enc_hs_pad(Tensor): 
                padded encoder hidden state (B, T_max, D_enc)
            enc_hs_len(list): 
                padded encoder hidden state length (B,)
            dec_z(Tensor): 
                decoder hidden state (B, D_dec)
            att_prev(Tensor): 
                attention weights of previous step (B, T_max)
            scaling(float, optional): 
                scaling parameter before applying softmax (Default value = 1.0)
            last_attended_idx(int, optional): 
                index of the inputs of the last attended (Default value = None)
            backward_window(int, optional): 
                backward window size in attention constraint (Default value = 1)
            forward_window(int, optional):  
                (Default value = 3)

        Returns:
            Tensor: 
                attention weighted encoder state (B, D_enc)
            Tensor: 
                previous attention weights (B, T_max)
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = paddle.shape(self.enc_h)[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = paddle.zeros([batch, self.dunits])
        else:
            dec_z = dec_z.reshape([batch, self.dunits])

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = paddle.zeros([*paddle.shape(enc_hs_pad)[:2]])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.reshape([batch, 1, 1, self.h_length]))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose([0, 2, 1])
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).unsqueeze(1)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(
            paddle.tanh(self.pre_compute_enc_h + dec_z_tiled +
                        att_conv)).squeeze(2)

        # NOTE: consider zero padding when compute w.
        if self.mask is None:
            self.mask = make_pad_mask(enc_hs_len)
        e = masked_fill(e, self.mask, -float("inf"))

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(e, last_attended_idx,
                                            backward_window, forward_window)

        w = F.softmax(scaling * e, axis=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (0, 0, 1, 0))[:, :-1]

        w = (att_prev + att_prev_shift) * w
        # NOTE: clip is needed to avoid nan gradient
        w = F.normalize(paddle.clip(w, 1e-6), p=1, axis=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = paddle.sum(self.enc_h * w.unsqueeze(-1), axis=1)

        return c, w


class AttForwardTA(nn.Layer):
    """Forward attention with transition agent module.
    Reference:
        Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
            (https://arxiv.org/pdf/1807.06736.pdf)

    Args:
        eunits (int): 
            units of encoder
        dunits (int): 
            units of decoder
        att_dim (int): 
            attention dimension
        aconv_chans (int):  
            channels of attention convolution
        aconv_filts (int): 
            filter size of attention convolution
        odim (int): 
            output dimension
    """

    def __init__(self, eunits, dunits, att_dim, aconv_chans, aconv_filts, odim):
        super().__init__()
        self.mlp_enc = nn.Linear(eunits, att_dim)
        self.mlp_dec = nn.Linear(dunits, att_dim, bias_attr=False)
        self.mlp_ta = nn.Linear(eunits + dunits + odim, 1)
        self.mlp_att = nn.Linear(aconv_chans, att_dim, bias_attr=False)
        self.loc_conv = nn.Conv2D(
            1,
            aconv_chans,
            (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias_attr=False, )
        self.gvec = nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eunits = eunits
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def reset(self):
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def forward(
            self,
            enc_hs_pad,
            enc_hs_len,
            dec_z,
            att_prev,
            out_prev,
            scaling=1.0,
            last_attended_idx=None,
            backward_window=1,
            forward_window=3, ):
        """Calculate AttForwardTA forward propagation.

        Args:
            enc_hs_pad(Tensor): 
                padded encoder hidden state (B, Tmax, eunits)
            enc_hs_len(list Tensor): 
                padded encoder hidden state length (B,)
            dec_z(Tensor): 
                decoder hidden state (B, dunits)
            att_prev(Tensor): 
                attention weights of previous step (B, T_max)
            out_prev(Tensor): 
                decoder outputs of previous step (B, odim)
            scaling(float, optional): 
                scaling parameter before applying softmax (Default value = 1.0)
            last_attended_idx(int, optional): 
                index of the inputs of the last attended (Default value = None)
            backward_window(int, optional): 
                backward window size in attention constraint (Default value = 1)
            forward_window(int, optional):  
                (Default value = 3)

        Returns:
            Tensor: 
                attention weighted encoder state (B, dunits)
            Tensor: 
                previous attention weights (B, Tmax)
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = paddle.shape(self.enc_h)[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = paddle.zeros([batch, self.dunits])
        else:
            dec_z = dec_z.reshape([batch, self.dunits])

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = paddle.zeros([*paddle.shape(enc_hs_pad)[:2]])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.reshape([batch, 1, 1, self.h_length]))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose([0, 2, 1])
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).reshape([batch, 1, self.att_dim])

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(
            paddle.tanh(att_conv + self.pre_compute_enc_h +
                        dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = make_pad_mask(enc_hs_len)
        e = masked_fill(e, self.mask, -float("inf"))

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(e, last_attended_idx,
                                            backward_window, forward_window)

        w = F.softmax(scaling * e, axis=1)

        # forward attention
        # att_prev_shift = F.pad(att_prev.unsqueeze(0), (1, 0), data_format='NCL').squeeze(0)[:, :-1]
        att_prev_shift = F.pad(att_prev, (0, 0, 1, 0))[:, :-1]
        w = (self.trans_agent_prob * att_prev +
             (1 - self.trans_agent_prob) * att_prev_shift) * w
        # NOTE: clip is needed to avoid nan gradient
        w = F.normalize(paddle.clip(w, 1e-6), p=1, axis=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = paddle.sum(
            self.enc_h * w.reshape([batch, self.h_length, 1]), axis=1)

        # update transition agent prob
        self.trans_agent_prob = F.sigmoid(
            self.mlp_ta(paddle.concat([c, out_prev, dec_z], axis=1)))

        return c, w
