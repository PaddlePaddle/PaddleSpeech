"""Contains DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.v2 as paddle


def conv_bn_layer(input, filter_size, num_channels_in, num_channels_out, stride,
                  padding, act):
    """
    Convolution layer with batch normalization.
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
    """
    Bidirectonal simple rnn layer with sequence-wise batch normalization.
    The batch normalization is only performed on input-state weights.
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


def conv_group(input, num_stacks):
    """
    Convolution group with several stacking convolution layers.
    """
    conv = conv_bn_layer(
        input=input,
        filter_size=(11, 41),
        num_channels_in=1,
        num_channels_out=32,
        stride=(3, 2),
        padding=(5, 20),
        act=paddle.activation.BRelu())
    for i in xrange(num_stacks - 1):
        conv = conv_bn_layer(
            input=conv,
            filter_size=(11, 21),
            num_channels_in=32,
            num_channels_out=32,
            stride=(1, 2),
            padding=(5, 10),
            act=paddle.activation.BRelu())
    output_num_channels = 32
    output_height = 160 // pow(2, num_stacks) + 1
    return conv, output_num_channels, output_height


def rnn_group(input, size, num_stacks):
    """
    RNN group with several stacking RNN layers.
    """
    output = input
    for i in xrange(num_stacks):
        output = bidirectional_simple_rnn_bn_layer(
            name=str(i), input=output, size=size, act=paddle.activation.BRelu())
    return output


def deep_speech2(audio_data,
                 text_data,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=256,
                 is_inference=False):
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
    :param is_inference: False in the training mode, and True in the
                         inferene mode.
    :type is_inference: bool
    :return: If is_inference set False, return a ctc cost layer;
             if is_inference set True, return a sequence layer of output
             probability distribution.
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
        input=conv2seq, size=rnn_size, num_stacks=num_rnn_layers)
    fc = paddle.layer.fc(
        input=rnn_group_output,
        size=dict_size + 1,
        act=paddle.activation.Linear(),
        bias_attr=True)
    if is_inference:
        # probability distribution with softmax
        return paddle.layer.mixed(
            input=paddle.layer.identity_projection(input=fc),
            act=paddle.activation.Softmax())
    else:
        # ctc cost
        return paddle.layer.warp_ctc(
            input=fc,
            label=text_data,
            size=dict_size + 1,
            blank=dict_size,
            norm_by_times=True)
