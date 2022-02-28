"""CNN Transformer model for SE in the SpeechBrain style.

Authors
* Chien-Feng Liao 2020
"""
import paddle  # noqa E402
from torch import nn
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
)


class CNNTransformerSE(TransformerInterface):
    """This is an implementation of transformer model with CNN pre-encoder for SE.

    Arguments
    ---------
    d_model : int
        The number of expected features in the encoder inputs.
    output_size : int
        The number of neurons in the output layer.
    output_activation : torch class
        The activation function of the output layer (default=ReLU).
    nhead : int
        The number of heads in the multi-head attention models (default=8).
    num_layers : int
        The number of sub-layers in the transformer (default=8).
    d_ffn : int
        The number of expected features in the encoder layers (default=512).
    dropout : int
        The dropout value (default=0.1).
    activation : torch class
        The activation function of intermediate layers (default=LeakyReLU).
    causal : bool
        True for causal setting, the model is forbidden to see future frames (default=True).
    custom_emb_module : torch class
        Module that processes the input features before the transformer model.

    Example
    -------
    >>> src = torch.rand([8, 120, 256])
    >>> net = CNNTransformerSE(d_model=256, output_size=257)
    >>> out = net(src)
    >>> out.shape
    torch.Size([8, 120, 257])
    """

    def __init__(
        self,
        d_model,
        output_size,
        output_activation=nn.ReLU,
        nhead=8,
        num_layers=8,
        d_ffn=512,
        dropout=0.1,
        activation=nn.LeakyReLU,
        causal=True,
        custom_emb_module=None,
        normalize_before=False,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=None,
            normalize_before=normalize_before,
            causal=causal,
        )

        self.custom_emb_module = custom_emb_module
        self.output_layer = Linear(output_size, input_size=d_model, bias=False)
        self.output_activation = output_activation()

    def forward(self, x, src_key_padding_mask=None):
        if self.causal:
            self.attn_mask = get_lookahead_mask(x)
        else:
            self.attn_mask = None

        if self.custom_emb_module is not None:
            x = self.custom_emb_module(x)

        encoder_output, _ = self.encoder(
            src=x,
            src_mask=self.attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        output = self.output_layer(encoder_output)
        output = self.output_activation(output)

        return output
