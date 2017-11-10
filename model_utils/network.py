"""Contains DeepSpeech2 layers and networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.v2 as paddle


def conv_bn_layer(input, filter_size, num_channels_in, num_channels_out, stride,
                  padding, act, index_range_data):
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
    :param index_range_data: Index range to indicate sub region.
    :type index_range_data: LayerOutput
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
    batch_norm = paddle.layer.batch_norm(input=conv_layer, act=act)
    # reset padding part to 0
    scale_sub_region = paddle.layer.scale_sub_region(
        batch_norm, index_range_data, value=0.0)
    return scale_sub_region


def bidirectional_simple_rnn_bn_layer(name, input, size, act, share_weights):
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
    :param share_weights: Whether to share input-hidden weights between
                          forward and backward directional RNNs.
    :type share_weights: bool
    :return: Bidirectional simple rnn layer.
    :rtype: LayerOutput
    """
    if share_weights:
        # input-hidden weights shared between bi-direcitonal rnn.
        input_proj = paddle.layer.fc(
            input=input,
            size=size,
            act=paddle.activation.Linear(),
            bias_attr=False)
        # batch norm is only performed on input-state projection
        input_proj_bn = paddle.layer.batch_norm(
            input=input_proj, act=paddle.activation.Linear())
        # forward and backward in time
        forward_simple_rnn = paddle.layer.recurrent(
            input=input_proj_bn, act=act, reverse=False)
        backward_simple_rnn = paddle.layer.recurrent(
            input=input_proj_bn, act=act, reverse=True)

    else:
        input_proj_forward = paddle.layer.fc(
            input=input,
            size=size,
            act=paddle.activation.Linear(),
            bias_attr=False)
        input_proj_backward = paddle.layer.fc(
            input=input,
            size=size,
            act=paddle.activation.Linear(),
            bias_attr=False)
        # batch norm is only performed on input-state projection
        input_proj_bn_forward = paddle.layer.batch_norm(
            input=input_proj_forward, act=paddle.activation.Linear())
        input_proj_bn_backward = paddle.layer.batch_norm(
            input=input_proj_backward, act=paddle.activation.Linear())
        # forward and backward in time
        forward_simple_rnn = paddle.layer.recurrent(
            input=input_proj_bn_forward, act=act, reverse=False)
        backward_simple_rnn = paddle.layer.recurrent(
            input=input_proj_bn_backward, act=act, reverse=True)

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
    input_proj_forward = paddle.layer.fc(
        input=input,
        size=size * 3,
        act=paddle.activation.Linear(),
        bias_attr=False)
    input_proj_backward = paddle.layer.fc(
        input=input,
        size=size * 3,
        act=paddle.activation.Linear(),
        bias_attr=False)
    # batch norm is only performed on input-related projections
    input_proj_bn_forward = paddle.layer.batch_norm(
        input=input_proj_forward, act=paddle.activation.Linear())
    input_proj_bn_backward = paddle.layer.batch_norm(
        input=input_proj_backward, act=paddle.activation.Linear())
    # forward and backward in time
    forward_gru = paddle.layer.grumemory(
        input=input_proj_bn_forward, act=act, reverse=False)
    backward_gru = paddle.layer.grumemory(
        input=input_proj_bn_backward, act=act, reverse=True)
    return paddle.layer.concat(input=[forward_gru, backward_gru])


def conv_group(input, num_stacks, index_range_datas):
    """Convolution group with stacked convolution layers.

    :param input: Input layer.
    :type input: LayerOutput
    :param num_stacks: Number of stacked convolution layers.
    :type num_stacks: int
    :param index_range_datas: Index ranges for each convolution layer.
    :type index_range_datas: tuple|list
    :return: Output layer of the convolution group.
    :rtype: LayerOutput
    """
    conv = conv_bn_layer(
        input=input,
        filter_size=(11, 41),
        num_channels_in=1,
        num_channels_out=32,
        stride=(3, 2),
        padding=(5, 20),
        act=paddle.activation.BRelu(),
        index_range_data=index_range_datas[0])
    for i in xrange(num_stacks - 1):
        conv = conv_bn_layer(
            input=conv,
            filter_size=(11, 21),
            num_channels_in=32,
            num_channels_out=32,
            stride=(1, 2),
            padding=(5, 10),
            act=paddle.activation.BRelu(),
            index_range_data=index_range_datas[i + 1])
    output_num_channels = 32
    output_height = 160 // pow(2, num_stacks) + 1
    return conv, output_num_channels, output_height


def rnn_group(input, size, num_stacks, use_gru, share_rnn_weights):
    """RNN group with stacked bidirectional simple RNN layers.

    :param input: Input layer.
    :type input: LayerOutput
    :param size: Number of RNN cells in each layer.
    :type size: int
    :param num_stacks: Number of stacked rnn layers.
    :type num_stacks: int
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward directional RNNs.
                              It is only available when use_gru=False.
    :type share_weights: bool
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
            # BRelu does not support hppl, need to add later. Use Relu instead.
        else:
            output = bidirectional_simple_rnn_bn_layer(
                name=str(i),
                input=output,
                size=size,
                act=paddle.activation.BRelu(),
                share_weights=share_rnn_weights)
    return output


def deep_speech_v2_network(audio_data,
                           text_data,
                           seq_offset_data,
                           seq_len_data,
                           index_range_datas,
                           dict_size,
                           num_conv_layers=2,
                           num_rnn_layers=3,
                           rnn_size=256,
                           use_gru=False,
                           share_rnn_weights=True):
    """The DeepSpeech2 network structure.

    :param audio_data: Audio spectrogram data layer.
    :type audio_data: LayerOutput
    :param text_data: Transcription text data layer.
    :type text_data: LayerOutput
    :param seq_offset_data: Sequence offset data layer.
    :type seq_offset_data: LayerOutput
    :param seq_len_data: Valid sequence length data layer.
    :type seq_len_data: LayerOutput
    :param index_range_datas: Index ranges data layers.
    :type index_range_datas: tuple|list
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
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward direction RNNs.
                              It is only available when use_gru=False.
    :type share_weights: bool
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """
    # convolution group
    conv_group_output, conv_group_num_channels, conv_group_height = conv_group(
        input=audio_data,
        num_stacks=num_conv_layers,
        index_range_datas=index_range_datas)
    # convert data form convolution feature map to sequence of vectors
    conv2seq = paddle.layer.block_expand(
        input=conv_group_output,
        num_channels=conv_group_num_channels,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=conv_group_height)
    # remove padding part
    remove_padding_data = paddle.layer.sub_seq(
        input=conv2seq,
        offsets=seq_offset_data,
        sizes=seq_len_data,
        act=paddle.activation.Linear(),
        bias_attr=False)
    # rnn group
    rnn_group_output = rnn_group(
        input=remove_padding_data,
        size=rnn_size,
        num_stacks=num_rnn_layers,
        use_gru=use_gru,
        share_rnn_weights=share_rnn_weights)
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
