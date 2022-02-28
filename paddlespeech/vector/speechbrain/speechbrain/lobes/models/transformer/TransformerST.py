"""Transformer for ST in the SpeechBrain sytle.

Authors
* YAO FEI, CHENG 2021
"""

import paddle  # noqa 42
import logging
from torch import nn
from typing import Optional

from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.Transformer import (
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
    TransformerDecoder,
    TransformerEncoder,
)
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.nnet.activations import Swish

logger = logging.getLogger(__name__)


class TransformerST(TransformerASR):
    """This is an implementation of transformer model for ST.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int, optional
        The number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int, optional
        The dimension of the feedforward network model (default=2048).
    dropout : int, optional
        The dropout value (default=0.1).
    activation : paddle.nn.Layer, optional
        The activation function of FFN layers.
        Recommended: relu or gelu (default=relu).
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        Choose between Conformer and Transformer for the encoder. The decoder is fixed to be a Transformer.
    conformer_activation: paddle.nn.Layer, optional
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    ctc_weight: float
        The weight of ctc for asr task
    asr_weight: float
        The weight of asr task for calculating loss
    mt_weight: float
        The weight of mt task for calculating loss
    asr_tgt_vocab: int
        The size of the asr target language
    mt_src_vocab: int
        The size of the mt source language
    Example
    -------
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerST(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU,
    ...     ctc_weight=1, asr_weight=0.3,
    ... )
    >>> enc_out, dec_out = net.forward(src, tgt)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    >>> dec_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Layer] = Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        ctc_weight: float = 0.0,
        asr_weight: float = 0.0,
        mt_weight: float = 0.0,
        asr_tgt_vocab: int = 0,
        mt_src_vocab: int = 0,
    ):
        super().__init__(
            tgt_vocab=tgt_vocab,
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
        )

        if ctc_weight < 1 and asr_weight > 0:
            self.asr_decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=True,
                attention_type="regularMHA",  # always use regular attention in decoder
            )
            self.custom_asr_tgt_module = ModuleList(
                NormalizedEmbedding(d_model, asr_tgt_vocab)
            )

        if mt_weight > 0:
            self.custom_mt_src_module = ModuleList(
                NormalizedEmbedding(d_model, mt_src_vocab)
            )
            if encoder_module == "transformer":
                self.mt_encoder = TransformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=self.causal,
                    attention_type=self.attention_type,
                )
            elif encoder_module == "conformer":
                self.mt_encoder = ConformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=conformer_activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=self.causal,
                    attention_type=self.attention_type,
                )
                assert (
                    normalize_before
                ), "normalize_before must be True for Conformer"

                assert (
                    conformer_activation is not None
                ), "conformer_activation must not be None"

        # reset parameters using xavier_normal_
        self._init_params()

    def forward_asr(self, encoder_out, src, tgt, wav_len, pad_idx=0):
        """This method implements a decoding step for asr task

        Arguments
        ----------
        encoder_out : tensor
            The representation of the encoder (required).
        tgt (transcription): tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, wav_len, pad_idx=pad_idx)

        transcription = self.custom_asr_tgt_module(tgt)

        if self.attention_type == "RelPosMHAXL":
            transcription = transcription + self.positional_encoding_decoder(
                transcription
            )
        elif self.attention_type == "fixed_abs_sine":
            transcription = transcription + self.positional_encoding(
                transcription
            )

        asr_decoder_out, _, _ = self.asr_decoder(
            tgt=transcription,
            memory=encoder_out,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return asr_decoder_out

    def forward_mt(self, src, tgt, pad_idx=0):
        """This method implements a forward step for mt task

        Arguments
        ----------
        src (transcription): tensor
            The sequence to the encoder (required).
        tgt (translation): tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks_for_mt(src, tgt, pad_idx=pad_idx)

        src = self.custom_mt_src_module(src)

        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)
            pos_embs_encoder = None

        encoder_out, _ = self.mt_encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        tgt = self.custom_tgt_module(tgt)

        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            src = src + self.positional_encoding_decoder(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return encoder_out, decoder_out

    def decode_asr(self, tgt, encoder_out):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : paddle.Tensor
            The sequence to the decoder.
        encoder_out : paddle.Tensor
            Hidden output of the encoder.
        """
        tgt_mask = get_lookahead_mask(tgt)
        tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            # we use fixed positional encodings in the decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            encoder_out = encoder_out + self.positional_encoding_decoder(
                encoder_out
            )
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)  # add the encodings here

        prediction, _, multihead_attns = self.asr_decoder(
            tgt, encoder_out, tgt_mask=tgt_mask,
        )

        return prediction, multihead_attns[-1]

    def make_masks_for_mt(self, src, tgt, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if self.training:
            src_key_padding_mask = get_key_padding_mask(src, pad_idx=pad_idx)
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)

        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask
