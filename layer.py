"""Contains DeepSpeech2 layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.v2 as paddle


def conv_bn_layer(input, filter_size, num_channels_in, num_channels_out, stride,
                  padding, act):
    """Convolution layer with batch normalization.

    :param input: Input layer.
    :type input: LayerOutput
    :param filter_size: The x dimension of a filter kernel. Or input a tuple for
                        two image dimension.
    :type filter_size: int|tuple|list
    :param num_channels_in: Number of input channels.
    :type num_channels_in: int
    :type num_channels_out: Number of output channels.
    :type num_channels_in: out
    :param padding: The x dimension of the padding. Or input a tuple for two
                    image dimension.
    :type padding: int|tuple|list
    :param act: Activation type.
    :type act: BaseActivation
    :return: Batch norm layer after convolution layer.
    :rtype: LayerOutput
    """
    conv_layer = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=num_channels_in,
        num_filters=num_channels_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=conv_layer, act=act)


def bidirectional_simple_rnn_bn_layer(name, input, size, act):
    """Bidirectonal simple rnn layer with sequence-wise batch normalization.
    The batch normalization is only performed on input-state weights.

    :param name: Name of the layer.
    :type name: string
    :param input: Input layer.
    :type input: LayerOutput
    :param size: Number of RNN cells.
    :type size: int
    :param act: Activation type.
    :type act: BaseActivation
    :return: Bidirectional simple rnn layer.
    :rtype: LayerOutput
    """
    # input-hidden weights shared across bi-direcitonal rnn.
    input_proj = paddle.layer.fc(
        input=input, size=size, act=paddle.activation.Linear(), bias_attr=False)
    # batch norm is only performed on input-state projection
    input_proj_bn = paddle.layer.batch_norm(
        input=input_proj, act=paddle.activation.Linear())
    # forward and backward in time
    forward_simple_rnn = paddle.layer.recurrent(
        input=input_proj_bn, act=act, reverse=False)
    backward_simple_rnn = paddle.layer.recurrent(
        input=input_proj_bn, act=act, reverse=True)
    return paddle.layer.concat(input=[forward_simple_rnn, backward_simple_rnn])


def bidirectional_gru_bn_layer(name, input, size, act):
    """Bidirectonal gru layer with sequence-wise batch normalization.
    The batch normalization is only performed on input-state weights.

    :param name: Name of the layer.
    :type name: string
    :param input: Input layer.
    :type input: LayerOutput
    :param size: Number of RNN cells.
    :type size: int
    :param act: Activation type.
    :type act: BaseActivation
    :return: Bidirectional simple rnn layer.
    :rtype: LayerOutput
    """
    # input-hidden weights shared across bi-direcitonal rnn.
    input_proj = paddle.layer.fc(
        input=input,
        size=size * 3,
        act=paddle.activation.Linear(),
        bias_attr=False)
    # batch norm is only performed on input-state projection
    input_proj_bn = paddle.layer.batch_norm(
        input=input_proj, act=paddle.activation.Linear())
    # forward and backward in time
    forward_gru = paddle.layer.grumemory(
        input=input_proj_bn, act=act, reverse=False)
    backward_gru = paddle.layer.grumemory(
        input=input_proj_bn, act=act, reverse=True)
    return paddle.layer.concat(input=[forward_gru, backward_gru])


def conv_group(input, num_stacks):
    """Convolution group with stacked convolution layers.

    :param input: Input layer.
    :type input: LayerOutput
    :param num_stacks: Number of stacked convolution layers.
    :type num_stacks: int
    :return: Output layer of the convolution group.
    :rtype: LayerOutput
    """
    conv = conv_bn_layer(
        input=input,
        filter_size=(11, 41),
        num_channels_in=1,
        num_channels_out=32,
        stride=(2, 2),
        padding=(5, 20),
        act=paddle.activation.Relu())
    for i in xrange(num_stacks - 1):
        conv = conv_bn_layer(
            input=conv,
            filter_size=(11, 21),
            num_channels_in=32,
            num_channels_out=32,
            stride=(1, 2),
            padding=(5, 10),
            act=paddle.activation.Relu())
    output_num_channels = 32
    output_height = 160 // pow(2, num_stacks) + 1
    return conv, output_num_channels, output_height


def rnn_group(input, size, num_stacks, use_gru):
    """RNN group with stacked bidirectional simple RNN layers.

    :param input: Input layer.
    :type input: LayerOutput
    :param size: Number of RNN cells in each layer.
    :type size: int
    :param num_stacks: Number of stacked rnn layers.
    :type num_stacks: int
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :return: Output layer of the RNN group.
    :rtype: LayerOutput
    """
    output = input
    for i in xrange(num_stacks):
        if use_gru:
            output = bidirectional_gru_bn_layer(
                name=str(i),
                input=output,
                size=size,
                act=paddle.activation.Relu())
        else:
            output = bidirectional_simple_rnn_bn_layer(
                name=str(i),
                input=output,
                size=size,
                act=paddle.activation.Relu())
    return output


def deep_speech2(audio_data,
                 text_data,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=256,
                 use_gru=True):
    """
    The whole DeepSpeech2 model structure (a simplified version).

    :param audio_data: Audio spectrogram data layer.
    :type audio_data: LayerOutput
    :param text_data: Transcription text data layer.
    :type text_data: LayerOutput
    :param dict_size: Dictionary size for tokenized transcription.
    :type dict_size: int
    :param num_conv_layers: Number of stacking convolution layers.
    :type num_conv_layers: int
    :param num_rnn_layers: Number of stacking RNN layers.
    :type num_rnn_layers: int
    :param rnn_size: RNN layer size (number of RNN cells).
    :type rnn_size: int
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """
    # convolution group
    conv_group_output, conv_group_num_channels, conv_group_height = conv_group(
        input=audio_data, num_stacks=num_conv_layers)
    # convert data form convolution feature map to sequence of vectors
    conv2seq = paddle.layer.block_expand(
        input=conv_group_output,
        num_channels=conv_group_num_channels,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=conv_group_height)
    # rnn group
    rnn_group_output = rnn_group(
        input=conv2seq,
        size=rnn_size,
        num_stacks=num_rnn_layers,
        use_gru=use_gru)
    fc = paddle.layer.fc(
        input=rnn_group_output,
        size=dict_size + 1,
        act=paddle.activation.Linear(),
        bias_attr=True)
    # probability distribution with softmax
    log_probs = paddle.layer.mixed(
        input=paddle.layer.identity_projection(input=fc),
        act=paddle.activation.Softmax())
    # ctc cost
    ctc_loss = paddle.layer.warp_ctc(
        input=fc,
        label=text_data,
        size=dict_size + 1,
        blank=dict_size,
        norm_by_times=True)
    return log_probs, ctc_loss
