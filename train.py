import paddle.v2 as paddle
import audio_data_utils
import argparse

parser = argparse.ArgumentParser(
    description='Simpled version of DeepSpeech2 trainer.')
parser.add_argument(
    "--batch_size", default=512, type=int, help="Minibatch size.")
parser.add_argument("--trainer", default=1, type=int, help="Trainer number.")
parser.add_argument(
    "--num_passes", default=20, type=int, help="Training pass number.")
args = parser.parse_args()


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


def conv_group(input):
    conv1 = conv_bn_layer(
        input=input,
        filter_size=(11, 41),
        num_channels_in=1,
        num_channels_out=32,
        stride=(3, 2),
        padding=(5, 20),
        act=paddle.activation.BRelu())
    conv2 = conv_bn_layer(
        input=conv1,
        filter_size=(11, 21),
        num_channels_in=32,
        num_channels_out=32,
        stride=(1, 2),
        padding=(5, 10),
        act=paddle.activation.BRelu())
    conv3 = conv_bn_layer(
        input=conv2,
        filter_size=(11, 21),
        num_channels_in=32,
        num_channels_out=32,
        stride=(1, 2),
        padding=(5, 10),
        act=paddle.activation.BRelu())
    return conv3


def rnn_group(input, size, num_stacks):
    output = input
    for i in xrange(num_stacks):
        output = bidirectonal_simple_rnn_bn_layer(
            name=str(i), input=output, size=size, act=paddle.activation.BRelu())
    return output


def deep_speech2(audio_data, text_data, dict_size):
    conv_group_output = conv_group(input=audio_data)
    conv2seq = paddle.layer.block_expand(
        input=conv_group_output,
        num_channels=32,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=21)
    rnn_group_output = rnn_group(input=conv2seq, size=256, num_stacks=5)
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
    return cost


def train():
    # create network config
    dict_size = audio_data_utils.get_vocabulary_size()
    audio_data = paddle.layer.data(
        name="audio_spectrogram",
        height=161,
        width=1000,
        type=paddle.data_type.dense_vector(161000))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(dict_size))
    cost = deep_speech2(audio_data, text_data, dict_size)

    # create parameters and optimizer
    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Adam(
        learning_rate=5e-5,
        gradient_clipping_threshold=5,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)
    return

    # create data readers
    feeding = {
        "audio_spectrogram": 0,
        "transcript_text": 1,
    }
    train_batch_reader = audio_data_utils.padding_batch_reader(
        paddle.batch(
            audio_data_utils.reader_creator("./libri.manifest.dev"),
            batch_size=args.batch_size // args.trainer),
        padding=[-1, 1000])
    test_batch_reader = audio_data_utils.padding_batch_reader(
        paddle.batch(
            audio_data_utils.reader_creator("./libri.manifest.test"),
            batch_size=args.batch_size // args.trainer),
        padding=[-1, 1000])

    # create event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print "Pass: %d, Batch: %d, TrainCost: %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_batch_reader, feeding=feeding)
            print "Pass: %d, TestCost: %f, %s" % (event.pass_id, event.cost,
                                                  result.metrics)
            with gzip.open("params.tar.gz", 'w') as f:
                parameters.to_tar(f)

    # run train
    trainer.train(
        reader=train_batch_reader,
        event_handler=event_handler,
        num_passes=10,
        feeding=feeding)


def main():
    train()


if __name__ == '__main__':
    main()
