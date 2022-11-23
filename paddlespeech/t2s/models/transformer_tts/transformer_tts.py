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
"""Fastspeech2 related modules for paddle"""
from typing import Dict
from typing import Sequence
from typing import Tuple

import numpy
import paddle
import paddle.nn.functional as F
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
from paddlespeech.t2s.modules.nets_utils import make_pad_mask
from paddlespeech.t2s.modules.style_encoder import StyleEncoder
from paddlespeech.t2s.modules.tacotron2.decoder import Postnet
from paddlespeech.t2s.modules.tacotron2.decoder import Prenet as DecoderPrenet
from paddlespeech.t2s.modules.tacotron2.encoder import Encoder as EncoderPrenet
from paddlespeech.t2s.modules.transformer.attention import MultiHeadedAttention
from paddlespeech.t2s.modules.transformer.decoder import Decoder
from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import ScaledPositionalEncoding
from paddlespeech.t2s.modules.transformer.encoder import TransformerEncoder
from paddlespeech.t2s.modules.transformer.mask import subsequent_mask


class TransformerTTS(nn.Layer):
    """TTS-Transformer module.

    This is a module of text-to-speech Transformer described in `Neural Speech Synthesis
    with Transformer Network`_, which convert the sequence of tokens into the sequence
    of Mel-filterbanks.

    .. _`Neural Speech Synthesis with Transformer Network`:
        https://arxiv.org/pdf/1809.08895.pdf

    Args:
        idim (int): 
            Dimension of the inputs.
        odim (int): 
            Dimension of the outputs.
        embed_dim (int, optional): 
            Dimension of character embedding.
        eprenet_conv_layers (int, optional): 
            Number of encoder prenet convolution layers.
        eprenet_conv_chans (int, optional): 
            Number of encoder prenet convolution channels.
        eprenet_conv_filts (int, optional): 
            Filter size of encoder prenet convolution.
        dprenet_layers (int, optional): 
            Number of decoder prenet layers.
        dprenet_units (int, optional): 
            Number of decoder prenet hidden units.
        elayers (int, optional): 
            Number of encoder layers.
        eunits (int, optional): 
            Number of encoder hidden units.
        adim (int, optional): 
            Number of attention transformation dimensions.
        aheads (int, optional): 
            Number of heads for multi head attention.
        dlayers (int, optional): 
            Number of decoder layers.
        dunits (int, optional): 
            Number of decoder hidden units.
        postnet_layers (int, optional): 
            Number of postnet layers.
        postnet_chans (int, optional): 
            Number of postnet channels.
        postnet_filts (int, optional): 
            Filter size of postnet.
        use_scaled_pos_enc (pool, optional): 
            Whether to use trainable scaled positional encoding.
        use_batch_norm (bool, optional): 
            Whether to use batch normalization in encoder prenet.
        encoder_normalize_before (bool, optional): 
            Whether to perform layer normalization before encoder block.
        decoder_normalize_before (bool, optional): 
            Whether to perform layer normalization before decoder block.
        encoder_concat_after (bool, optional): 
            Whether to concatenate attention layer's input and output in encoder.
        decoder_concat_after (bool, optional): 
            Whether to concatenate attention layer's input and output in decoder.
        positionwise_layer_type (str, optional): 
            Position-wise operation type.
        positionwise_conv_kernel_size (int, optional): 
            Kernel size in position wise conv 1d.
        reduction_factor (int, optional): 
            Reduction factor.
        spk_embed_dim (int, optional): 
            Number of speaker embedding dimenstions.
        spk_embed_integration_type (str, optional): 
            How to integrate speaker embedding.
        use_gst (str, optional): 
            Whether to use global style token.
        gst_tokens (int, optional): 
            The number of GST embeddings.
        gst_heads (int, optional): 
            The number of heads in GST multihead attention.
        gst_conv_layers (int, optional): 
            The number of conv layers in GST.
        gst_conv_chans_list (Sequence[int], optional): 
            List of the number of channels of conv layers in GST.
        gst_conv_kernel_size (int, optional): 
            Kernal size of conv layers in GST.
        gst_conv_stride (int, optional): 
            Stride size of conv layers in GST.
        gst_gru_layers (int, optional): 
            The number of GRU layers in GST.
        gst_gru_units (int, optional): 
            The number of GRU units in GST.
        transformer_lr (float, optional): 
            Initial value of learning rate.
        transformer_warmup_steps (int, optional): 
            Optimizer warmup steps.
        transformer_enc_dropout_rate (float, optional): 
            Dropout rate in encoder except attention and positional encoding.
        transformer_enc_positional_dropout_rate (float, optional): 
            Dropout rate after encoder positional encoding.
        transformer_enc_attn_dropout_rate （float, optional): 
            Dropout rate in encoder self-attention module.
        transformer_dec_dropout_rate (float, optional): 
            Dropout rate in decoder except attention & positional encoding.
        transformer_dec_positional_dropout_rate (float, optional): 
            Dropout rate after decoder positional encoding.
        transformer_dec_attn_dropout_rate （float, optional): 
            Dropout rate in deocoder self-attention module.
        transformer_enc_dec_attn_dropout_rate (float, optional): 
            Dropout rate in encoder-deocoder attention module.
        init_type (str, optional): 
            How to initialize transformer parameters.
        init_enc_alpha （float, optional）: 
            Initial value of alpha in scaled pos encoding of the encoder.
        init_dec_alpha (float, optional): 
            Initial value of alpha in scaled pos encoding of the decoder.
        eprenet_dropout_rate (float, optional): 
            Dropout rate in encoder prenet.
        dprenet_dropout_rate (float, optional): 
            Dropout rate in decoder prenet.
        postnet_dropout_rate (float, optional): 
            Dropout rate in postnet.
        use_masking (bool, optional): 
            Whether to apply masking for padded part in loss calculation.
        use_weighted_masking (bool, optional): 
            Whether to apply weighted masking in loss calculation.
        bce_pos_weight (float, optional): 
            Positive sample weight in bce calculation (only for use_masking=true).
        loss_type (str, optional): 
            How to calculate loss.
        use_guided_attn_loss (bool, optional): 
            Whether to use guided attention loss.
        num_heads_applied_guided_attn (int, optional):
            Number of heads in each layer to apply guided attention loss.
        num_layers_applied_guided_attn (int, optional): 
            Number of layers to apply guided attention loss.
    """

    def __init__(
            self,
            # network structure related
            idim: int,
            odim: int,
            embed_dim: int=512,
            eprenet_conv_layers: int=3,
            eprenet_conv_chans: int=256,
            eprenet_conv_filts: int=5,
            dprenet_layers: int=2,
            dprenet_units: int=256,
            elayers: int=6,
            eunits: int=1024,
            adim: int=512,
            aheads: int=4,
            dlayers: int=6,
            dunits: int=1024,
            postnet_layers: int=5,
            postnet_chans: int=256,
            postnet_filts: int=5,
            positionwise_layer_type: str="conv1d",
            positionwise_conv_kernel_size: int=1,
            use_scaled_pos_enc: bool=True,
            use_batch_norm: bool=True,
            encoder_normalize_before: bool=True,
            decoder_normalize_before: bool=True,
            encoder_concat_after: bool=False,
            decoder_concat_after: bool=False,
            reduction_factor: int=1,
            spk_embed_dim: int=None,
            spk_embed_integration_type: str="add",
            use_gst: bool=False,
            gst_tokens: int=10,
            gst_heads: int=4,
            gst_conv_layers: int=6,
            gst_conv_chans_list: Sequence[int]=(32, 32, 64, 64, 128, 128),
            gst_conv_kernel_size: int=3,
            gst_conv_stride: int=2,
            gst_gru_layers: int=1,
            gst_gru_units: int=128,
            # training related
            transformer_enc_dropout_rate: float=0.1,
            transformer_enc_positional_dropout_rate: float=0.1,
            transformer_enc_attn_dropout_rate: float=0.1,
            transformer_dec_dropout_rate: float=0.1,
            transformer_dec_positional_dropout_rate: float=0.1,
            transformer_dec_attn_dropout_rate: float=0.1,
            transformer_enc_dec_attn_dropout_rate: float=0.1,
            eprenet_dropout_rate: float=0.5,
            dprenet_dropout_rate: float=0.5,
            postnet_dropout_rate: float=0.5,
            init_type: str="xavier_uniform",
            init_enc_alpha: float=1.0,
            init_dec_alpha: float=1.0,
            use_guided_attn_loss: bool=True,
            num_heads_applied_guided_attn: int=2,
            num_layers_applied_guided_attn: int=2, ):
        """Initialize Transformer module."""
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
        self.reduction_factor = reduction_factor
        self.use_gst = use_gst
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.use_guided_attn_loss = use_guided_attn_loss
        if self.use_guided_attn_loss:
            if num_layers_applied_guided_attn == -1:
                self.num_layers_applied_guided_attn = elayers
            else:
                self.num_layers_applied_guided_attn = num_layers_applied_guided_attn
            if num_heads_applied_guided_attn == -1:
                self.num_heads_applied_guided_attn = aheads
            else:
                self.num_heads_applied_guided_attn = num_heads_applied_guided_attn
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0
        # set_global_initializer 会影响后面的全局，包括 create_parameter
        initialize(self, init_type)

        # get positional encoding layer type
        transformer_pos_enc_layer_type = "scaled_abs_pos" if self.use_scaled_pos_enc else "abs_pos"

        # define transformer encoder
        if eprenet_conv_layers != 0:
            # encoder prenet
            encoder_input_layer = nn.Sequential(
                EncoderPrenet(
                    idim=idim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx, ),
                nn.Linear(eprenet_conv_chans, adim), )
        else:
            encoder_input_layer = nn.Embedding(
                num_embeddings=idim,
                embedding_dim=adim,
                padding_idx=self.padding_idx)
        self.encoder = TransformerEncoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            pos_enc_layer_type=transformer_pos_enc_layer_type,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size, )

        # define GST
        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=adim,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units, )

        # define projection layer
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = nn.Linear(adim + self.spk_embed_dim, adim)

        # define transformer decoder
        if dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = nn.Sequential(
                DecoderPrenet(
                    idim=odim,
                    n_layers=dprenet_layers,
                    n_units=dprenet_units,
                    dropout_rate=dprenet_dropout_rate, ),
                nn.Linear(dprenet_units, adim), )
        else:
            decoder_input_layer = "linear"
        # get positional encoding class
        pos_enc_class = (ScaledPositionalEncoding
                         if self.use_scaled_pos_enc else PositionalEncoding)
        self.decoder = Decoder(
            odim=odim,  # odim is needed when no prenet is used
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            self_attention_dropout_rate=transformer_dec_attn_dropout_rate,
            src_attention_dropout_rate=transformer_enc_dec_attn_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=pos_enc_class,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after, )

        # define final projection
        self.feat_out = nn.Linear(adim, odim * reduction_factor)
        self.prob_out = nn.Linear(adim, reduction_factor)

        # define postnet
        self.postnet = (None if postnet_layers == 0 else Postnet(
            idim=idim,
            odim=odim,
            n_layers=postnet_layers,
            n_chans=postnet_chans,
            n_filts=postnet_filts,
            use_batch_norm=use_batch_norm,
            dropout_rate=postnet_dropout_rate, ))

        # 闭合的 initialize() 中的 set_global_initializer 的作用域，防止其影响到 self._reset_parameters()
        nn.initializer.set_global_initializer(None)

        self._reset_parameters(
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha, )

    def _reset_parameters(self, init_enc_alpha: float, init_dec_alpha: float):

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            init_enc_alpha = paddle.to_tensor(init_enc_alpha)
            self.encoder.embed[-1].alpha = paddle.create_parameter(
                shape=init_enc_alpha.shape,
                dtype=str(init_enc_alpha.numpy().dtype),
                default_initializer=paddle.nn.initializer.Assign(
                    init_enc_alpha))

            init_dec_alpha = paddle.to_tensor(init_dec_alpha)
            self.decoder.embed[-1].alpha = paddle.create_parameter(
                shape=init_dec_alpha.shape,
                dtype=str(init_dec_alpha.numpy().dtype),
                default_initializer=paddle.nn.initializer.Assign(
                    init_dec_alpha))

    def forward(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            spk_emb: paddle.Tensor=None,
    ) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor], paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            text(Tensor(int64)): Batch of padded character ids (B, Tmax).
            text_lengths(Tensor(int64)): Batch of lengths of each input batch (B,).
            speech(Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths(Tensor(int64)): Batch of the lengths of each target (B,).
            spk_emb(Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.

        """
        # input of embedding must be int64
        text_lengths = paddle.cast(text_lengths, 'int64')

        # Add eos at the last of sequence
        text = numpy.pad(text.numpy(), ((0, 0), (0, 1)), 'constant')
        xs = paddle.to_tensor(text, dtype='int64')
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = speech
        olens = paddle.cast(speech_lengths, 'int64')

        # make labels for stop prediction
        stop_labels = make_pad_mask(olens - 1)
        # bool 类型无法切片
        stop_labels = paddle.cast(stop_labels, dtype='float32')
        stop_labels = F.pad(stop_labels, [0, 0, 0, 1], "constant", 1.0)

        # calculate transformer outputs
        after_outs, before_outs, logits = self._forward(xs, ilens, ys, olens,
                                                        spk_emb)

        # modifiy mod part of groundtruth

        if self.reduction_factor > 1:
            olens = olens - olens % self.reduction_factor
            max_olen = max(olens)
            ys = ys[:, :max_olen]
            stop_labels = stop_labels[:, :max_olen]
            stop_labels[:, -1] = 1.0  # make sure at least one frame has 1
            olens_in = olens // self.reduction_factor
        else:
            olens_in = olens

        need_dict = {}
        need_dict['encoder'] = self.encoder
        need_dict['decoder'] = self.decoder
        need_dict[
            'num_heads_applied_guided_attn'] = self.num_heads_applied_guided_attn
        need_dict[
            'num_layers_applied_guided_attn'] = self.num_layers_applied_guided_attn
        need_dict['use_scaled_pos_enc'] = self.use_scaled_pos_enc

        return after_outs, before_outs, logits, ys, stop_labels, olens, olens_in, need_dict

    def _forward(
            self,
            xs: paddle.Tensor,
            ilens: paddle.Tensor,
            ys: paddle.Tensor,
            olens: paddle.Tensor,
            spk_emb: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, h_masks = self.encoder(xs, x_masks)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spk_emb)

        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.reduction_factor > 1:
            ys_in = ys[:, self.reduction_factor - 1::self.reduction_factor]
            olens_in = olens // self.reduction_factor
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # forward decoder
        y_masks = self._target_mask(olens_in)
        zs, _ = self.decoder(ys_in, y_masks, hs, h_masks)
        # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
        before_outs = self.feat_out(zs).reshape([zs.shape[0], -1, self.odim])
        # (B, Lmax//r, r) -> (B, Lmax//r * r)
        logits = self.prob_out(zs).reshape([zs.shape[0], -1])

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose([0, 2, 1])).transpose([0, 2, 1])

        return after_outs, before_outs, logits

    def inference(
            self,
            text: paddle.Tensor,
            speech: paddle.Tensor=None,
            spk_emb: paddle.Tensor=None,
            threshold: float=0.5,
            minlenratio: float=0.0,
            maxlenratio: float=10.0,
            use_teacher_forcing: bool=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text(Tensor(int64)): Input sequence of characters (T,).
            speech(Tensor, optional): Feature sequence to extract style (N, idim).
            spk_emb(Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            threshold(float, optional): Threshold in inference.
            minlenratio(float, optional): Minimum length ratio in inference.
            maxlenratio(float, optional): Maximum length ratio in inference.
            use_teacher_forcing(bool, optional): Whether to use teacher forcing.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).

        """
        # input of embedding must be int64
        y = speech

        # add eos at the last of sequence
        text = numpy.pad(
            text.numpy(), (0, 1), 'constant', constant_values=self.eos)
        x = paddle.to_tensor(text, dtype='int64')

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech is not None, "speech must be provided with teacher forcing."

            # get teacher forcing outputs
            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            spk_emb = None if spk_emb is None else spk_emb.unsqueeze(0)
            ilens = paddle.to_tensor(
                [xs.shape[1]], dtype=paddle.int64, place=xs.place)
            olens = paddle.to_tensor(
                [ys.shape[1]], dtype=paddle.int64, place=ys.place)
            outs, *_ = self._forward(xs, ilens, ys, olens, spk_emb)

            # get attention weights
            att_ws = []
            for i in range(len(self.decoder.decoders)):
                att_ws += [self.decoder.decoders[i].src_attn.attn]
            # (B, L, H, T_out, T_in)
            att_ws = paddle.stack(att_ws, axis=1)

            return outs[0], None, att_ws[0]

        # forward encoder
        xs = x.unsqueeze(0)
        hs, _ = self.encoder(xs, None)

        # integrate GST
        if self.use_gst:
            style_embs = self.gst(y.unsqueeze(0))
            hs = hs + style_embs.unsqueeze(1)

        # integrate speaker embedding
        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(0)
            hs = self._integrate_with_spk_embed(hs, spk_emb)

        # set limits of length
        maxlen = int(hs.shape[1] * maxlenratio / self.reduction_factor)
        minlen = int(hs.shape[1] * minlenratio / self.reduction_factor)

        # initialize
        idx = 0
        ys = paddle.zeros([1, 1, self.odim])
        outs, probs = [], []

        # forward decoder step-by-step
        z_cache = None
        while True:
            # update index
            idx += 1

            # calculate output and stop prob at idx-th step
            y_masks = subsequent_mask(idx).unsqueeze(0)
            z, z_cache = self.decoder.forward_one_step(
                ys, y_masks, hs, cache=z_cache)  # (B, adim)
            outs += [
                self.feat_out(z).reshape([self.reduction_factor, self.odim])
            ]  # [(r, odim), ...]
            probs += [F.sigmoid(self.prob_out(z))[0]]  # [(r), ...]

            # update next inputs
            ys = paddle.concat(
                (ys, outs[-1][-1].reshape([1, 1, self.odim])),
                axis=1)  # (1, idx + 1, odim)

            # get attention weights
            att_ws_ = []
            for name, m in self.named_sublayers():
                if isinstance(m, MultiHeadedAttention) and "src" in name:
                    # [(#heads, 1, T),...]
                    att_ws_ += [m.attn[0, :, -1].unsqueeze(1)]
            if idx == 1:
                att_ws = att_ws_
            else:
                # [(#heads, l, T), ...]
                att_ws = [
                    paddle.concat([att_w, att_w_], axis=1)
                    for att_w, att_w_ in zip(att_ws, att_ws_)
                ]

            # check whether to finish generation
            if sum(paddle.cast(probs[-1] >= threshold,
                               'int64')) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                # (L, odim) -> (1, L, odim) -> (1, odim, L)
                outs = (paddle.concat(outs, axis=0).unsqueeze(0).transpose(
                    [0, 2, 1]))
                if self.postnet is not None:
                    # (1, odim, L)
                    outs = outs + self.postnet(outs)
                # (L, odim)
                outs = outs.transpose([0, 2, 1]).squeeze(0)
                probs = paddle.concat(probs, axis=0)
                break

        # concatenate attention weights -> (#layers, #heads, L, T)
        att_ws = paddle.stack(att_ws, axis=0)

        return outs, probs, att_ws

    def _add_first_frame_and_remove_last_frame(
            self, ys: paddle.Tensor) -> paddle.Tensor:
        ys_in = paddle.concat(
            [paddle.zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], axis=1)
        return ys_in

    def _source_mask(self, ilens: paddle.Tensor) -> paddle.Tensor:
        """Make masks for self-attention.

        Args:
            ilens(Tensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention. dtype=paddle.bool

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 0]]]) bool

        """
        x_masks = make_non_pad_mask(ilens)
        return x_masks.unsqueeze(-2)

    def _target_mask(self, olens: paddle.Tensor) -> paddle.Tensor:
        """Make masks for masked self-attention.

        Args:
            olens (Tensor(int64)): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked self-attention.

        Examples:
            >>> olens = [5, 3]
            >>> self._target_mask(olens)
            tensor([[[1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1]],
                    [[1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0]]], dtype=paddle.uint8)

        """
        y_masks = make_non_pad_mask(olens)
        s_masks = subsequent_mask(y_masks.shape[-1]).unsqueeze(0)
        return paddle.logical_and(y_masks.unsqueeze(-2), s_masks)

    def _integrate_with_spk_embed(self,
                                  hs: paddle.Tensor,
                                  spk_emb: paddle.Tensor) -> paddle.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs(Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spk_emb(Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spk_emb = self.projection(F.normalize(spk_emb))
            hs = hs + spk_emb.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spk_emb = F.normalize(spk_emb).unsqueeze(1).expand(-1, hs.shape[1],
                                                               -1)
            hs = self.projection(paddle.concat([hs, spk_emb], axis=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs


class TransformerTTSInference(nn.Layer):
    def __init__(self, normalizer, model):
        super().__init__()
        self.normalizer = normalizer
        self.acoustic_model = model

    def forward(self, text, spk_id=None):
        normalized_mel = self.acoustic_model.inference(text)[0]
        logmel = self.normalizer.inverse(normalized_mel)
        return logmel
