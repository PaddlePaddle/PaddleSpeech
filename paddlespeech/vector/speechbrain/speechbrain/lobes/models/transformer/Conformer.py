"""Conformer implementation.

Authors
* Jianyuan Zhong 2020
* Samuele Cornell 2021
"""

import paddle
import paddle.nn as nn
from typing import Optional
import speechbrain as sb
import warnings


from speechbrain.nnet.attention import (
    RelPosMHAXL,
    MultiheadAttention,
    PositionalwiseFeedForward,
)
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.activations import Swish


class ConvolutionModule(nn.Layer):
    """This is an implementation of convolution module in Conformer.

    Arguments
    ----------
    input_size : int
        The expected size of the input embedding dimension.
    kernel_size: int, optional
        Kernel size of non-bottleneck convolutional layer.
    bias: bool, optional
        Whether to use bias in the non-bottleneck conv layer.
    activation: paddle.nn.Layer
         Activation function used after non-bottleneck conv layer.
    dropout: float, optional
         Dropout rate.
    causal: bool, optional
         Whether the convolution should be causal or not.
    dilation: int, optional
         Dilation factor for the non bottleneck conv layer.

    Example
    -------
    >>> import paddle
    >>> x = torch.rand((8, 60, 512))
    >>> net = ConvolutionModule(512, 3)
    >>> output = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        input_size,
        kernel_size=31,
        bias=True,
        activation=Swish,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.causal = causal

        if self.causal:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1)
        else:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1) // 2

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            # pointwise
            nn.Conv1d(
                input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.GLU(dim=1),
        )
        # depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        self.after_conv = nn.Sequential(
            nn.BatchNorm1d(input_size),
            activation(),
            # pointwise
            nn.Conv1d(
                input_size, input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.layer_norm(x)
        out = out.transpose(1, 2)
        out = self.bottleneck(out)
        out = self.conv(out)

        if self.causal:
            # chomp
            out = out[..., : -self.padding]
        out = self.after_conv(out)
        out = out.transpose(1, 2)
        return out


class ConformerEncoderLayer(nn.Layer):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ----------
    d_model : int
        The expected size of the input embedding.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: paddle.nn.Layer
         Activation function used in each Conformer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.

    Example
    -------
    >>> import paddle
    >>> x = torch.rand((8, 60, 512))
    >>> pos_embs = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoderLayer(d_ffn=512, nhead=8, d_model=512, kernel_size=3)
    >>> output = net(x, pos_embs=pos_embs)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
        elif attention_type == "RelPosMHAXL":
            # transformerXL style positional encoding
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=dropout,
                mask_pos_future=causal,
            )

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        src_mask: Optional[paddle.Tensor] = None,
        src_key_padding_mask: Optional[paddle.Tensor] = None,
        pos_embs: Optional[paddle.Tensor] = None,
    ):

        """
                Arguments
                ----------
                src : paddle.Tensor
                    The sequence to the encoder layer.
                src_mask : paddle.Tensor, optional
                    The mask for the src sequence.
                src_key_padding_mask : paddle.Tensor, optional
                    The mask for the src keys per batch.
                pos_embs: paddle.Tensor, paddle.nn.Layer, optional
                    Module or tensor containing the input sequence positional embeddings
                """
        # ffn module
        x = x + 0.5 * self.ffn_module1(x)
        # muti-head attention module
        skip = x
        x = self.norm1(x)
        x, self_attn = self.mha_layer(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )
        x = x + skip
        # convolution module
        x = x + self.convolution_module(x)
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn


class ConformerEncoder(nn.Layer):
    """This class implements the Conformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Embedding dimension size.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: paddle.nn.Layer
         Activation function used in each Confomer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.


    Example
    -------
    >>> import paddle
    >>> x = torch.rand((8, 60, 512))
    >>> pos_emb = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoder(1, 512, 512, 8)
    >>> output, _ = net(x, pos_embs=pos_emb)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[paddle.Tensor] = None,
        src_key_padding_mask: Optional[paddle.Tensor] = None,
        pos_embs: Optional[paddle.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : paddle.Tensor
            The sequence to the encoder layer.
        src_mask : paddle.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : paddle.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: paddle.Tensor, paddle.nn.Layer, optional
            Module or tensor containing the input sequence positional embeddings
        """
        output = src
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst


class ConformerDecoderLayer(nn.Layer):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ----------
    d_model : int
        The expected size of the input embedding.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: paddle.nn.Layer, optional
         Activation function used in each Conformer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.

    Example
    -------
    >>> import paddle
    >>> x = torch.rand((8, 60, 512))
    >>> pos_embs = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoderLayer(d_ffn=512, nhead=8, d_model=512, kernel_size=3)
    >>> output = net(x, pos_embs=pos_embs)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=True,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if not causal:
            warnings.warn(
                "Decoder is not causal, in most applications it should be causal, you have been warned !"
            )

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
        elif attention_type == "RelPosMHAXL":
            # transformerXL style positional encoding
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=dropout,
                mask_pos_future=causal,
            )

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
            tgt: paddle.Tensor
                The sequence to the decoder layer.
            memory: paddle.Tensor
                The sequence from the last layer of the encoder.
            tgt_mask: paddle.Tensor, optional, optional
                The mask for the tgt sequence.
            memory_mask: paddle.Tensor, optional
                The mask for the memory sequence.
            tgt_key_padding_mask : paddle.Tensor, optional
                The mask for the tgt keys per batch.
            memory_key_padding_mask : paddle.Tensor, optional
                The mask for the memory keys per batch.
            pos_emb_tgt: paddle.Tensor, paddle.nn.Layer, optional
                Module or tensor containing the target sequence positional embeddings for each attention layer.
            pos_embs_src: paddle.Tensor, paddle.nn.Layer, optional
                Module or tensor containing the source sequence positional embeddings for each attention layer.
        """
        # ffn module
        tgt = tgt + 0.5 * self.ffn_module1(tgt)
        # muti-head attention module
        skip = tgt
        x = self.norm1(tgt)
        x, self_attn = self.mha_layer(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            pos_embs=pos_embs_src,
        )
        x = x + skip
        # convolution module
        x = x + self.convolution_module(x)
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn, self_attn


class ConformerDecoder(nn.Layer):
    """This class implements the Transformer decoder.

    Arguments
    ----------
    num_layers: int
        Number of layers.
    nhead: int
        Number of attention heads.
    d_ffn: int
        Hidden size of self-attention Feed Forward layer.
    d_model: int
        Embedding dimension size.
    kdim: int, optional
        Dimension for key.
    vdim: int, optional
        Dimension for value.
    dropout: float, optional
        Dropout rate.
    activation: paddle.nn.Layer, optional
         Activation function used after non-bottleneck conv layer.
    kernel_size : int, optional
        Kernel size of convolutional layer.
    bias : bool, optional
        Whether  convolution module.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.


    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = ConformerDecoder(1, 8, 1024, 512, attention_type="regularMHA")
    >>> output, _, _ = net(tgt, src)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=Swish,
        kernel_size=3,
        bias=True,
        causal=True,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                ConformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt: paddle.Tensor
            The sequence to the decoder layer.
        memory: paddle.Tensor
            The sequence from the last layer of the encoder.
        tgt_mask: paddle.Tensor, optional, optional
            The mask for the tgt sequence.
        memory_mask: paddle.Tensor, optional
            The mask for the memory sequence.
        tgt_key_padding_mask : paddle.Tensor, optional
            The mask for the tgt keys per batch.
        memory_key_padding_mask : paddle.Tensor, optional
            The mask for the memory keys per batch.
        pos_emb_tgt: paddle.Tensor, paddle.nn.Layer, optional
            Module or tensor containing the target sequence positional embeddings for each attention layer.
        pos_embs_src: paddle.Tensor, paddle.nn.Layer, optional
            Module or tensor containing the source sequence positional embeddings for each attention layer.

        """
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        return output, self_attns, multihead_attns
