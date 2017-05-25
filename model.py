"""
   A simplifed version of Baidu DeepSpeech2 model.
"""

import paddle.v2 as paddle

#TODO: add bidirectional rnn.


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


def bidirectonal_simple_rnn_bn_layer(name, input, size, act):
    """
    Bidirectonal simple rnn layer with batch normalization.
    The batch normalization is only performed on input-state projection
    (sequence-wise normalization).

    Question: does mean and variance statistics computed over the whole sequence
    or just on each individual time steps?
    """

    def __simple_rnn_step__(input):
        last_state = paddle.layer.memory(name=name + "_state", size=size)
        input_fc = paddle.layer.fc(
            input=input,
            size=size,
            act=paddle.activation.Linear(),
            bias_attr=False)
        # batch norm is only performed on input-state projection 
        input_fc_bn = paddle.layer.batch_norm(
            input=input_fc, act=paddle.activation.Linear())
        state_fc = paddle.layer.fc(
            input=last_state,
            size=size,
            act=paddle.activation.Linear(),
            bias_attr=False)
        return paddle.layer.addto(
            name=name + "_state", input=[input_fc_bn, state_fc], act=act)

    forward = paddle.layer.recurrent_group(
        step=__simple_rnn_step__, input=input)
    return forward
    # argument reverse is not exposed in V2 recurrent_group
    #backward = paddle.layer.recurrent_group(


#step=__simple_rnn_step__,
#input=input,
#reverse=True)
#return paddle.layer.concat(input=[forward, backward])


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
    return conv


def rnn_group(input, size, num_stacks):
    """
    RNN group with several stacking RNN layers.
    """
    output = input
    for i in xrange(num_stacks):
        output = bidirectonal_simple_rnn_bn_layer(
            name=str(i), input=output, size=size, act=paddle.activation.BRelu())
    return output


def deep_speech2(audio_data,
                 text_data,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=256):
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
    :return: Tuple of the cost layer and the max_id decoder layer.
    :rtype: tuple of LayerOutput
    """
    # convolution group
    conv_group_output = conv_group(input=audio_data, num_stacks=num_conv_layers)
    # convert data form convolution feature map to sequence of vectors
    conv2seq = paddle.layer.block_expand(
        input=conv_group_output,
        num_channels=32,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=21)
    # rnn group
    rnn_group_output = rnn_group(
        input=conv2seq, size=rnn_size, num_stacks=num_rnn_layers)
    # output token distribution
    fc = paddle.layer.fc(
        input=rnn_group_output,
        size=dict_size + 1,
        act=paddle.activation.Linear(),
        bias_attr=True)
    # ctc cost
    cost = paddle.layer.warp_ctc(
        input=fc,
        label=text_data,
        size=dict_size + 1,
        blank=dict_size,
        norm_by_times=True)
    # max decoder
    max_id = paddle.layer.max_id(input=fc)
    return cost, max_id
