# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Tacotron 2 related modules for paddle"""
import logging
from typing import Dict
from typing import Optional
from typing import Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.nets_utils import make_pad_mask
from paddlespeech.t2s.modules.tacotron2.attentions import AttForward
from paddlespeech.t2s.modules.tacotron2.attentions import AttForwardTA
from paddlespeech.t2s.modules.tacotron2.attentions import AttLoc
from paddlespeech.t2s.modules.tacotron2.decoder import Decoder
from paddlespeech.t2s.modules.tacotron2.encoder import Encoder


class Tacotron2(nn.Layer):
    """Tacotron2 module for end-to-end text-to-speech.

    This is a module of Spectrogram prediction network in Tacotron2 described
    in `Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_,
    which converts the sequence of characters into the sequence of Mel-filterbanks.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
            self,
            # network structure related
            idim: int,
            odim: int,
            embed_dim: int=512,
            elayers: int=1,
            eunits: int=512,
            econv_layers: int=3,
            econv_chans: int=512,
            econv_filts: int=5,
            atype: str="location",
            adim: int=512,
            aconv_chans: int=32,
            aconv_filts: int=15,
            cumulate_att_w: bool=True,
            dlayers: int=2,
            dunits: int=1024,
            prenet_layers: int=2,
            prenet_units: int=256,
            postnet_layers: int=5,
            postnet_chans: int=512,
            postnet_filts: int=5,
            output_activation: str=None,
            use_batch_norm: bool=True,
            use_concate: bool=True,
            use_residual: bool=False,
            reduction_factor: int=1,
            # extra embedding related
            spk_num: Optional[int]=None,
            lang_num: Optional[int]=None,
            spk_embed_dim: Optional[int]=None,
            spk_embed_integration_type: str="concat",
            dropout_rate: float=0.5,
            zoneout_rate: float=0.1,
            # training related
            init_type: str="xavier_uniform", ):
        """Initialize Tacotron2 module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            embed_dim (int): Dimension of the token embedding.
            elayers (int): Number of encoder blstm layers.
            eunits (int): Number of encoder blstm units.
            econv_layers (int): Number of encoder conv layers.
            econv_filts (int): Number of encoder conv filter size.
            econv_chans (int): Number of encoder conv filter channels.
            dlayers (int): Number of decoder lstm layers.
            dunits (int): Number of decoder lstm units.
            prenet_layers (int): Number of prenet layers.
            prenet_units (int): Number of prenet units.
            postnet_layers (int): Number of postnet layers.
            postnet_filts (int): Number of postnet filter size.
            postnet_chans (int): Number of postnet filter channels.
            output_activation (str): Name of activation function for outputs.
            adim (int): Number of dimension of mlp in attention.
            aconv_chans (int): Number of attention conv filter channels.
            aconv_filts (int): Number of attention conv filter size.
            cumulate_att_w (bool): Whether to cumulate previous attention weight.
            use_batch_norm (bool): Whether to use batch normalization.
            use_concate (bool): Whether to concat enc outputs w/ dec lstm outputs.
            reduction_factor (int): Reduction factor.
            spk_num (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            lang_num (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spk_emb will be provided as the input.
            spk_embed_integration_type (str): How to integrate speaker embedding.
            dropout_rate (float): Dropout rate.
            zoneout_rate (float): Zoneout rate.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError(f"there is no such an activation function. "
                             f"({output_activation})")

        # set padding idx
        padding_idx = 0
        self.padding_idx = padding_idx

        # initialize parameters
        initialize(self, init_type)

        # define network modules
        self.enc = Encoder(
            idim=idim,
            embed_dim=embed_dim,
            elayers=elayers,
            eunits=eunits,
            econv_layers=econv_layers,
            econv_chans=econv_chans,
            econv_filts=econv_filts,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            dropout_rate=dropout_rate,
            padding_idx=padding_idx, )

        self.spk_num = None
        if spk_num is not None and spk_num > 1:
            self.spk_num = spk_num
            self.sid_emb = nn.Embedding(spk_num, eunits)
        self.lang_num = None
        if lang_num is not None and lang_num > 1:
            self.lang_num = lang_num
            self.lid_emb = nn.Embedding(lang_num, eunits)

        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is None:
            dec_idim = eunits
        elif self.spk_embed_integration_type == "concat":
            dec_idim = eunits + spk_embed_dim
        elif self.spk_embed_integration_type == "add":
            dec_idim = eunits
            self.projection = nn.Linear(self.spk_embed_dim, eunits)
        else:
            raise ValueError(f"{spk_embed_integration_type} is not supported.")

        if atype == "location":
            att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)
        elif atype == "forward":
            att = AttForward(dec_idim, dunits, adim, aconv_chans, aconv_filts)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled "
                                "in forward attention.")
                self.cumulate_att_w = False
        elif atype == "forward_ta":
            att = AttForwardTA(dec_idim, dunits, adim, aconv_chans, aconv_filts,
                               odim)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled "
                                "in forward attention.")
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        self.dec = Decoder(
            idim=dec_idim,
            odim=odim,
            att=att,
            dlayers=dlayers,
            dunits=dunits,
            prenet_layers=prenet_layers,
            prenet_units=prenet_units,
            postnet_layers=postnet_layers,
            postnet_chans=postnet_chans,
            postnet_filts=postnet_filts,
            output_activation_fn=self.output_activation_fn,
            cumulate_att_w=self.cumulate_att_w,
            use_batch_norm=use_batch_norm,
            use_concate=use_concate,
            dropout_rate=dropout_rate,
            zoneout_rate=zoneout_rate,
            reduction_factor=reduction_factor, )

        nn.initializer.set_global_initializer(None)

    def forward(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            spk_emb: Optional[paddle.Tensor]=None,
            spk_id: Optional[paddle.Tensor]=None,
            lang_id: Optional[paddle.Tensor]=None
    ) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor], paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            text (Tensor(int64)): Batch of padded character ids (B, T_text).
            text_lengths (Tensor(int64)): Batch of lengths of each input batch (B,).
            speech (Tensor): Batch of padded target features (B, T_feats, odim).
            speech_lengths (Tensor(int64)): Batch of the lengths of each target (B,).
            spk_emb (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            spk_id (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lang_id (Optional[Tensor]): Batch of language IDs (B, 1).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """
        text = text[:, :text_lengths.max()]
        speech = speech[:, :speech_lengths.max()]

        batch_size = paddle.shape(text)[0]

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 0, 0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = speech
        olens = speech_lengths

        # make labels for stop prediction
        stop_labels = make_pad_mask(olens - 1)
        # bool 类型无法切片
        stop_labels = paddle.cast(stop_labels, dtype='float32')
        stop_labels = F.pad(stop_labels, [0, 0, 0, 1], "constant", 1.0)

        # calculate tacotron2 outputs
        after_outs, before_outs, logits, att_ws = self._forward(
            xs=xs,
            ilens=ilens,
            ys=ys,
            olens=olens,
            spk_emb=spk_emb,
            spk_id=spk_id,
            lang_id=lang_id, )

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            assert olens.ge(self.reduction_factor).all(
            ), "Output length must be greater than or equal to reduction factor."
            olens = olens - olens % self.reduction_factor
            max_out = max(olens)
            ys = ys[:, :max_out]
            stop_labels = stop_labels[:, :max_out]
            stop_labels = paddle.scatter(stop_labels, 1,
                                         (olens - 1).unsqueeze(1), 1.0)
            olens_in = olens // self.reduction_factor
        else:
            olens_in = olens
        return after_outs, before_outs, logits, ys, stop_labels, olens, att_ws, olens_in

    def _forward(
            self,
            xs: paddle.Tensor,
            ilens: paddle.Tensor,
            ys: paddle.Tensor,
            olens: paddle.Tensor,
            spk_emb: paddle.Tensor,
            spk_id: paddle.Tensor,
            lang_id: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:

        hs, hlens = self.enc(xs, ilens)
        if self.spk_num is not None:
            sid_embs = self.sid_emb(spk_id.reshape([-1]))
            hs = hs + sid_embs.unsqueeze(1)
        if self.lang_num is not None:
            lid_embs = self.lid_emb(lang_id.reshape([-1]))
            hs = hs + lid_embs.unsqueeze(1)
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spk_emb)

        return self.dec(hs, hlens, ys)

    def inference(
            self,
            text: paddle.Tensor,
            speech: Optional[paddle.Tensor]=None,
            spk_emb: Optional[paddle.Tensor]=None,
            spk_id: Optional[paddle.Tensor]=None,
            lang_id: Optional[paddle.Tensor]=None,
            threshold: float=0.5,
            minlenratio: float=0.0,
            maxlenratio: float=10.0,
            use_att_constraint: bool=False,
            backward_window: int=1,
            forward_window: int=3,
            use_teacher_forcing: bool=False, ) -> Dict[str, paddle.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (Tensor(int64)): Input sequence of characters (T_text,).
            speech (Optional[Tensor]): Feature sequence to extract style (N, idim).
            spk_emb (ptional[Tensor]): Speaker embedding (spk_embed_dim,).
            spk_id (Optional[Tensor]): Speaker ID (1,).
            lang_id (Optional[Tensor]): Language ID (1,).
            threshold (float): Threshold in inference.
            minlenratio (float): Minimum length ratio in inference.
            maxlenratio (float): Maximum length ratio in inference.
            use_att_constraint (bool): Whether to apply attention constraint.
            backward_window (int): Backward window in attention constraint.
            forward_window (int): Forward window in attention constraint.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]
            Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * prob (Tensor): Output sequence of stop probabilities (T_feats,).
                * att_w (Tensor): Attention weights (T_feats, T).

        """
        x = text
        y = speech

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech is not None, "speech must be provided with teacher forcing."

            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            spk_emb = None if spk_emb is None else spk_emb.unsqueeze(0)
            ilens = paddle.shape(xs)[1]
            olens = paddle.shape(ys)[1]
            outs, _, _, att_ws = self._forward(
                xs=xs,
                ilens=ilens,
                ys=ys,
                olens=olens,
                spk_emb=spk_emb,
                spk_id=spk_id,
                lang_id=lang_id, )

            return dict(feat_gen=outs[0], att_w=att_ws[0])

        # inference
        h = self.enc.inference(x)

        if self.spk_num is not None:
            sid_emb = self.sid_emb(spk_id.reshape([-1]))
            h = h + sid_emb
        if self.lang_num is not None:
            lid_emb = self.lid_emb(lang_id.reshape([-1]))
            h = h + lid_emb
        if self.spk_embed_dim is not None:
            hs, spk_emb = h.unsqueeze(0), spk_emb.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, spk_emb)[0]
        out, prob, att_w = self.dec.inference(
            h,
            threshold=threshold,
            minlenratio=minlenratio,
            maxlenratio=maxlenratio,
            use_att_constraint=use_att_constraint,
            backward_window=backward_window,
            forward_window=forward_window, )

        return dict(feat_gen=out, prob=prob, att_w=att_w)

    def _integrate_with_spk_embed(self,
                                  hs: paddle.Tensor,
                                  spk_emb: paddle.Tensor) -> paddle.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, eunits).
            spk_emb (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, eunits) if
                integration_type is "add" else (B, Tmax, eunits + spk_embed_dim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spk_emb = self.projection(F.normalize(spk_emb))
            hs = hs + spk_emb.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            spk_emb = F.normalize(spk_emb).unsqueeze(1).expand(
                shape=[-1, paddle.shape(hs)[1], -1])
            hs = paddle.concat([hs, spk_emb], axis=-1)
        else:
            raise NotImplementedError("support only add or concat.")

        return hs


class Tacotron2Inference(nn.Layer):
    def __init__(self, normalizer, model):
        super().__init__()
        self.normalizer = normalizer
        self.acoustic_model = model

    def forward(self, text, spk_id=None, spk_emb=None):
        out = self.acoustic_model.inference(
            text, spk_id=spk_id, spk_emb=spk_emb)
        normalized_mel = out["feat_gen"]
        logmel = self.normalizer.inverse(normalized_mel)
        return logmel
