from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import paddle.fluid as fluid
import numpy as np


def conv_bn_layer(input, filter_size, num_channels_in, num_channels_out, stride,
                  padding, act, masks, name):
    """Convolution layer with batch normalization.

    :param input: Input layer.
    :type input: Variable
    :param filter_size: The x dimension of a filter kernel. Or input a tuple for
                        two image dimension.
    :type filter_size: int|tuple|list
    :param num_channels_in: Number of input channels.
    :type num_channels_in: int
    :param num_channels_out: Number of output channels.
    :type num_channels_out: int
    :param stride: The x dimension of the stride. Or input a tuple for two 
                   image dimension. 
    :type stride: int|tuple|list
    :param padding: The x dimension of the padding. Or input a tuple for two
                    image dimension.
    :type padding: int|tuple|list
    :param act: Activation type.
    :type act: string
    :param masks: Masks data layer to reset padding.
    :type masks: Variable
    :param name: Name of the layer.
    :param name: string
    :return: Batch norm layer after convolution layer.
    :rtype: Variable
   
    """
    conv_layer = fluid.layers.conv2d(
        input=input,
        num_filters=num_channels_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        param_attr=fluid.ParamAttr(name=name + '_conv2d_weight'),
        act=None,
        bias_attr=False)

    batch_norm = fluid.layers.batch_norm(
        input=conv_layer,
        act=act,
        param_attr=fluid.ParamAttr(name=name + '_batch_norm_weight'),
        bias_attr=fluid.ParamAttr(name=name + '_batch_norm_bias'),
        moving_mean_name=name + '_batch_norm_moving_mean',
        moving_variance_name=name + '_batch_norm_moving_variance')

    # reset padding part to 0
    padding_reset = fluid.layers.elementwise_mul(batch_norm, masks)
    return padding_reset


class RNNCell(fluid.layers.RNNCell):
    """A simple rnn cell."""

    def __init__(self,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 hidden_activation=None,
                 activation=None,
                 dtype="float32",
                 name="RNNCell"):
        """Initialize simple rnn cell.

        :param hidden_size: Dimension of RNN cells.
        :type hidden_size: int
        :param param_attr: Parameter properties of hidden layer weights that 
                      can be learned
        :type param_attr: ParamAttr
        :param bias_attr: Bias properties of hidden layer weights that can be learned
        :type bias_attr: ParamAttr
        :param hidden_activation: Activation for hidden cell
        :type hidden_activation: Activation
        :param activation: Activation for output
        :type activation: Activation
        :param name: Name of cell
        :type name: string
        """

        self.hidden_size = hidden_size
        self.param_attr = param_attr
        self.bias_attr = bias_attr
        self.hidden_activation = hidden_activation
        self.activation = activation or fluid.layers.brelu
        self.name = name

    def call(self, inputs, states):
        new_hidden = fluid.layers.fc(
            input=states,
            size=self.hidden_size,
            act=self.hidden_activation,
            param_attr=self.param_attr,
            bias_attr=self.bias_attr)
        new_hidden = fluid.layers.elementwise_add(new_hidden, inputs)
        new_hidden = self.activation(new_hidden)

        return new_hidden, new_hidden

    @property
    def state_shape(self):
        return [self.hidden_size]


def bidirectional_simple_rnn_bn_layer(name, input, size, share_weights):
    """Bidirectonal simple rnn layer with sequence-wise batch normalization.
    The batch normalization is only performed on input-state weights.

    :param name: Name of the layer parameters.
    :type name: string
    :param input: Input layer.
    :type input: Variable
    :param size: Dimension of RNN cells.
    :type size: int
    :param share_weights: Whether to share input-hidden weights between
                          forward and backward directional RNNs.
    :type share_weights: bool
    :return: Bidirectional simple rnn layer.
    :rtype: Variable
    """
    forward_cell = RNNCell(
        hidden_size=size,
        activation=fluid.layers.brelu,
        param_attr=fluid.ParamAttr(name=name + '_forward_rnn_weight'),
        bias_attr=fluid.ParamAttr(name=name + '_forward_rnn_bias'))

    reverse_cell = RNNCell(
        hidden_size=size,
        activation=fluid.layers.brelu,
        param_attr=fluid.ParamAttr(name=name + '_reverse_rnn_weight'),
        bias_attr=fluid.ParamAttr(name=name + '_reverse_rnn_bias'))

    pad_value = fluid.layers.assign(input=np.array([0.0], dtype=np.float32))

    if share_weights:
        #input-hidden weights shared between bi-directional rnn.
        input_proj = fluid.layers.fc(
            input=input,
            size=size,
            act=None,
            param_attr=fluid.ParamAttr(name=name + '_fc_weight'),
            bias_attr=False)

        # batch norm is only performed on input-state projection
        input_proj_bn_forward = fluid.layers.batch_norm(
            input=input_proj,
            act=None,
            param_attr=fluid.ParamAttr(name=name + '_batch_norm_weight'),
            bias_attr=fluid.ParamAttr(name=name + '_batch_norm_bias'),
            moving_mean_name=name + '_batch_norm_moving_mean',
            moving_variance_name=name + '_batch_norm_moving_variance')
        input_proj_bn_reverse = input_proj_bn_forward
    else:
        input_proj_forward = fluid.layers.fc(
            input=input,
            size=size,
            act=None,
            param_attr=fluid.ParamAttr(name=name + '_forward_fc_weight'),
            bias_attr=False)
        input_proj_reverse = fluid.layers.fc(
            input=input,
            size=size,
            act=None,
            param_attr=fluid.ParamAttr(name=name + '_reverse_fc_weight'),
            bias_attr=False)
        #batch norm is only performed on input-state projection
        input_proj_bn_forward = fluid.layers.batch_norm(
            input=input_proj_forward,
            act=None,
            param_attr=fluid.ParamAttr(
                name=name + '_forward_batch_norm_weight'),
            bias_attr=fluid.ParamAttr(name=name + '_forward_batch_norm_bias'),
            moving_mean_name=name + '_forward_batch_norm_moving_mean',
            moving_variance_name=name + '_forward_batch_norm_moving_variance')
        input_proj_bn_reverse = fluid.layers.batch_norm(
            input=input_proj_reverse,
            act=None,
            param_attr=fluid.ParamAttr(
                name=name + '_reverse_batch_norm_weight'),
            bias_attr=fluid.ParamAttr(name=name + '_reverse_batch_norm_bias'),
            moving_mean_name=name + '_reverse_batch_norm_moving_mean',
            moving_variance_name=name + '_reverse_batch_norm_moving_variance')
    # forward and backward in time
    input, length = fluid.layers.sequence_pad(input_proj_bn_forward, pad_value)
    forward_rnn, _ = fluid.layers.rnn(
        cell=forward_cell, inputs=input, time_major=False, is_reverse=False)
    forward_rnn = fluid.layers.sequence_unpad(x=forward_rnn, length=length)

    input, length = fluid.layers.sequence_pad(input_proj_bn_reverse, pad_value)
    reverse_rnn, _ = fluid.layers.rnn(
        cell=reverse_cell,
        inputs=input,
        sequence_length=length,
        time_major=False,
        is_reverse=True)
    reverse_rnn = fluid.layers.sequence_unpad(x=reverse_rnn, length=length)

    out = fluid.layers.concat(input=[forward_rnn, reverse_rnn], axis=1)
    return out


def bidirectional_gru_bn_layer(name, input, size, act):
    """Bidirectonal gru layer with sequence-wise batch normalization.
    The batch normalization is only performed on input-state weights.

    :param name: Name of the layer.
    :type name: string
    :param input: Input layer.
    :type input: Variable
    :param size: Dimension of GRU cells.
    :type size: int
    :param act: Activation type.
    :type act: string
    :return: Bidirectional GRU layer.
    :rtype: Variable
    """
    input_proj_forward = fluid.layers.fc(
        input=input,
        size=size * 3,
        act=None,
        param_attr=fluid.ParamAttr(name=name + '_forward_fc_weight'),
        bias_attr=False)
    input_proj_reverse = fluid.layers.fc(
        input=input,
        size=size * 3,
        act=None,
        param_attr=fluid.ParamAttr(name=name + '_reverse_fc_weight'),
        bias_attr=False)
    #batch norm is only performed on input-related prohections
    input_proj_bn_forward = fluid.layers.batch_norm(
        input=input_proj_forward,
        act=None,
        param_attr=fluid.ParamAttr(name=name + '_forward_batch_norm_weight'),
        bias_attr=fluid.ParamAttr(name=name + '_forward_batch_norm_bias'),
        moving_mean_name=name + '_forward_batch_norm_moving_mean',
        moving_variance_name=name + '_forward_batch_norm_moving_variance')
    input_proj_bn_reverse = fluid.layers.batch_norm(
        input=input_proj_reverse,
        act=None,
        param_attr=fluid.ParamAttr(name=name + '_reverse_batch_norm_weight'),
        bias_attr=fluid.ParamAttr(name=name + '_reverse_batch_norm_bias'),
        moving_mean_name=name + '_reverse_batch_norm_moving_mean',
        moving_variance_name=name + '_reverse_batch_norm_moving_variance')
    #forward and backward in time
    forward_gru = fluid.layers.dynamic_gru(
        input=input_proj_bn_forward,
        size=size,
        gate_activation='sigmoid',
        candidate_activation=act,
        param_attr=fluid.ParamAttr(name=name + '_forward_gru_weight'),
        bias_attr=fluid.ParamAttr(name=name + '_forward_gru_bias'),
        is_reverse=False)
    reverse_gru = fluid.layers.dynamic_gru(
        input=input_proj_bn_reverse,
        size=size,
        gate_activation='sigmoid',
        candidate_activation=act,
        param_attr=fluid.ParamAttr(name=name + '_reverse_gru_weight'),
        bias_attr=fluid.ParamAttr(name=name + '_reverse_gru_bias'),
        is_reverse=True)
    return fluid.layers.concat(input=[forward_gru, reverse_gru], axis=1)


def conv_group(input, num_stacks, seq_len_data, masks):
    """Convolution group with stacked convolution layers.

    :param input: Input layer.
    :type input: Variable
    :param num_stacks: Number of stacked convolution layers.
    :type num_stacks: int
    :param seq_len_data:Valid sequence length data layer. 
    :type seq_len_data:Variable
    :param masks: Masks data layer to reset padding.
    :type masks: Variable
    :return: Output layer of the convolution group.
    :rtype: Variable
    """
    filter_size = (41, 11)
    stride = (2, 3)
    padding = (20, 5)
    conv = conv_bn_layer(
        input=input,
        filter_size=filter_size,
        num_channels_in=1,
        num_channels_out=32,
        stride=stride,
        padding=padding,
        act="brelu",
        masks=masks,
        name='layer_0', )

    seq_len_data = (np.array(seq_len_data) - filter_size[1] + 2 * padding[1]
                    ) // stride[1] + 1

    output_height = (161 - 1) // 2 + 1

    for i in range(num_stacks - 1):
        #reshape masks
        output_height = (output_height - 1) // 2 + 1
        masks = fluid.layers.slice(
            masks, axes=[2], starts=[0], ends=[output_height])
        conv = conv_bn_layer(
            input=conv,
            filter_size=(21, 11),
            num_channels_in=32,
            num_channels_out=32,
            stride=(2, 1),
            padding=(10, 5),
            act="brelu",
            masks=masks,
            name='layer_{}'.format(i + 1), )

    output_num_channels = 32
    return conv, output_num_channels, output_height, seq_len_data


def rnn_group(input, size, num_stacks, num_conv_layers, use_gru,
              share_rnn_weights):
    """RNN group with stacked bidirectional simple RNN or GRU layers.

    :param input: Input layer.
    :type input: Variable
    :param size: Dimension of RNN cells in each layer.
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
    :rtype: Variable
    """
    output = input
    for i in range(num_stacks):
        if use_gru:
            output = bidirectional_gru_bn_layer(
                name='layer_{}'.format(i + num_conv_layers),
                input=output,
                size=size,
                act="relu")
        else:
            name = 'layer_{}'.format(i + num_conv_layers)
            output = bidirectional_simple_rnn_bn_layer(
                name=name,
                input=output,
                size=size,
                share_weights=share_rnn_weights)
    return output


def deep_speech_v2_network(audio_data,
                           text_data,
                           seq_len_data,
                           masks,
                           dict_size,
                           num_conv_layers=2,
                           num_rnn_layers=3,
                           rnn_size=256,
                           use_gru=False,
                           share_rnn_weights=True):
    """The DeepSpeech2 network structure.

    :param audio_data: Audio spectrogram data layer.
    :type audio_data: Variable
    :param text_data: Transcription text data layer.
    :type text_data: Variable
    :param seq_len_data: Valid sequence length data layer.
    :type seq_len_data: Variable
    :param masks: Masks data layer to reset padding.
    :type masks: Variable
    :param dict_size: Dictionary size for tokenized transcription.
    :type dict_size: int
    :param num_conv_layers: Number of stacking convolution layers.
    :type num_conv_layers: int
    :param num_rnn_layers: Number of stacking RNN layers.
    :type num_rnn_layers: int
    :param rnn_size: RNN layer size (dimension of RNN cells).
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
    audio_data = fluid.layers.unsqueeze(audio_data, axes=[1])

    # convolution group
    conv_group_output, conv_group_num_channels, conv_group_height, seq_len_data = conv_group(
        input=audio_data,
        num_stacks=num_conv_layers,
        seq_len_data=seq_len_data,
        masks=masks)

    # convert data form convolution feature map to sequence of vectors
    transpose = fluid.layers.transpose(conv_group_output, perm=[0, 3, 1, 2])
    reshape_conv_output = fluid.layers.reshape(
        x=transpose,
        shape=[0, -1, conv_group_height * conv_group_num_channels],
        inplace=False)
    # remove padding part
    seq_len_data = fluid.layers.reshape(seq_len_data, [-1])
    sequence = fluid.layers.sequence_unpad(
        x=reshape_conv_output, length=seq_len_data)
    #rnn group
    rnn_group_output = rnn_group(
        input=sequence,
        size=rnn_size,
        num_stacks=num_rnn_layers,
        num_conv_layers=num_conv_layers,
        use_gru=use_gru,
        share_rnn_weights=share_rnn_weights)
    fc = fluid.layers.fc(
        input=rnn_group_output,
        size=dict_size + 1,
        act=None,
        param_attr=fluid.ParamAttr(
            name='layer_{}'.format(num_conv_layers + num_rnn_layers) +
            '_fc_weight'),
        bias_attr=fluid.ParamAttr(
            name='layer_{}'.format(num_conv_layers + num_rnn_layers) +
            '_fc_bias'))
    # pribability distribution with softmax
    log_probs = fluid.layers.softmax(fc)
    log_probs.persistable = True
    if not text_data:
        return log_probs, None
    else:
        #ctc cost
        ctc_loss = fluid.layers.warpctc(
            input=fc, label=text_data, blank=dict_size, norm_by_times=True)
        ctc_loss = fluid.layers.reduce_sum(ctc_loss)
        return log_probs, ctc_loss
