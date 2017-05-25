import paddle.v2 as paddle


def conv_bn_layer(input, filter_size, num_channels_in, num_channels_out, stride,
                  padding, act):
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
    def __simple_rnn_step__(input):
        last_state = paddle.layer.memory(name=name + "_state", size=size)
        input_fc = paddle.layer.fc(
            input=input,
            size=size,
            act=paddle.activation.Linear(),
            bias_attr=False)
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
    conv_group_output = conv_group(input=audio_data, num_stacks=num_conv_layers)
    conv2seq = paddle.layer.block_expand(
        input=conv_group_output,
        num_channels=32,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=21)
    rnn_group_output = rnn_group(
        input=conv2seq, size=rnn_size, num_stacks=num_rnn_layers)
    fc = paddle.layer.fc(
        input=rnn_group_output,
        size=dict_size + 1,
        act=paddle.activation.Linear(),
        bias_attr=True)
    cost = paddle.layer.warp_ctc(
        input=fc,
        label=text_data,
        size=dict_size + 1,
        blank=dict_size,
        norm_by_times=True)
    max_id = paddle.layer.max_id(input=fc)
    return cost, max_id
