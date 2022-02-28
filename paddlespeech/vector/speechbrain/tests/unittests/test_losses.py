import paddle
import pytest


def test_nll(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.losses import nll_loss

    predictions = paddle.zeros([4, 10, 8], dtype="float32")
    targets = paddle.zeros([4, 10], dtype="float32")
    lengths = paddle.ones([4], dtype="float32")
    out_cost = nll_loss(predictions, targets, lengths)
    assert paddle.all(paddle.equal(out_cost, 0))


def test_mse(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.losses import mse_loss

    predictions = paddle.ones([4, 10, 8], )
    targets = paddle.ones([4, 10, 8], )
    lengths = paddle.ones([4], )
    out_cost = mse_loss(predictions, targets, lengths)
    assert paddle.all(paddle.equal(out_cost, 0))

    predictions = paddle.zeros([4, 10, 8], )
    out_cost = mse_loss(predictions, targets, lengths)
    assert paddle.all(paddle.equal(out_cost, 1))


def test_l1(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.losses import l1_loss

    predictions = paddle.ones([4, 10, 8], )
    targets = paddle.ones([4, 10, 8], )
    lengths = paddle.ones([4], )
    out_cost = l1_loss(predictions, targets, lengths)
    assert paddle.all(paddle.equal(out_cost, 0))


def test_bce_loss(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.losses import bce_loss

    # Ensure this works both with and without singleton dimension
    predictions_singleton = paddle.zeros([4, 10, 1], )
    predictions_match = paddle.zeros([4, 10], )
    targets = paddle.ones([4, 10], )
    lengths = paddle.ones([4], )
    out_cost_singleton = bce_loss(predictions_singleton, targets, lengths)
    out_cost_match = bce_loss(predictions_match, targets, lengths)
    assert paddle.allclose(
        paddle.exp(out_cost_singleton), paddle.to_tensor([2.0], )
    )
    assert paddle.allclose(
        paddle.exp(out_cost_match), paddle.to_tensor([2.0], )
    )

    # How about one dimensional inputs
    predictions = paddle.zeros([5, 1], )
    targets = paddle.ones([5], )
    out_cost = bce_loss(predictions, targets)
    assert paddle.allclose(paddle.exp(out_cost), paddle.to_tensor([2.0], ))

    # Can't pass lengths in 1D case
    with pytest.raises(ValueError):
        bce_loss(predictions, targets, length=paddle.ones([5], ))


def test_classification_error(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.losses import classification_error

    predictions = paddle.zeros([4, 10, 8], )
    predictions[:, :, 0] += 1.0
    targets = paddle.zeros([4, 10], )
    lengths = paddle.ones([4], )
    out_cost = classification_error(predictions, targets, lengths)
    assert paddle.all(paddle.equal(out_cost, 0))


def test_pitwrapper(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.losses import PitWrapper
    from paddle import nn

    base_loss = nn.MSELoss(reduction="none")
    pit = PitWrapper(base_loss)
    predictions = paddle.rand(
        [2, 32, 4], 
    )  # batch, frames, sources
    p = (3, 0, 2, 1)
    # same but we invert the ordering to check if permutation invariant
    # 目前 paddle中的Tensor不支持下面的获取数据方式
    targets = paddle.to_tensor(predictions.numpy()[..., p])
    loss, opt_p = pit(predictions, targets)
    assert [x == p for x in opt_p] == [True for i in range(len(opt_p))]
    predictions = pit.reorder_tensor(predictions, opt_p)
    assert paddle.all(paddle.equal(base_loss(predictions, targets), 0))

    predictions = paddle.rand(
        (3, 32, 32, 32, 5), 
    )  # batch, ..., sources
    p = (3, 0, 2, 1, 4)
    targets = paddle.to_tensor(predictions.numpy()[
        ..., p
    ])  # same but we invert the ordering to check if permutation invariant
    loss, opt_p = pit(predictions, targets)
    assert [x == p for x in opt_p] == [True for i in range(len(opt_p))]
    predictions = pit.reorder_tensor(predictions, opt_p)
    assert paddle.all(paddle.equal(base_loss(predictions, targets), 0))


# def test_transducer_loss(device):
#     # 放弃测试transducer loss
#     paddle.device.set_device(device)
#     # Make this its own test since it can only be run
#     # if numba is installed and a GPU is available
#     pytest.importorskip("numba")
#     if paddle.device.cuda.device_count() == 0:
#         pytest.skip("This test can only be run if a GPU is available")

#     from speechbrain.nnet.losses import transducer_loss

#     paddle.device.set_device("gpu")
#     device = paddle.device.get_device()
#     log_probs = paddle.nn.functional.log_softmax(
#         paddle.to_tensor(
#             [
#                 [
#                     [
#                         [0.1, 0.6, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.6, 0.1, 0.1],
#                         [0.1, 0.1, 0.2, 0.8, 0.1],
#                     ],
#                     [
#                         [0.1, 0.6, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.2, 0.1, 0.1],
#                         [0.7, 0.1, 0.2, 0.1, 0.1],
#                     ],
#                 ]
#             ]
#         , stop_gradient=False
#         ),
#         axis=-1
#     )
#             # .log_softmax(dim=-1)
#     # print("log_probs.place: {}".format(log_probs.place))
#     targets = paddle.to_tensor([[1, 2]]).astype("int32")
#     probs_length = paddle.to_tensor([1.0])
#     target_length = paddle.to_tensor([1.0])
#     out_cost = transducer_loss(
#         log_probs,
#         targets,
#         probs_length,
#         target_length,
#         blank_index=0,
#         use_torchaudio=False,
#     )
#     assert out_cost.item() == 2.247833251953125


# def test_guided_attention_loss_mask(device):
#     paddle.device.set_device(device)
#     from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss

#     loss = GuidedAttentionLoss().to(device)
#     input_lengths = paddle.tensor([3, 2, 6], )
#     output_lengths = paddle.tensor([4, 3, 5], )
#     soft_mask = loss.guided_attentions(input_lengths, output_lengths)
#     ref_soft_mask = paddle.tensor(
#         [
#             [
#                 [0.0, 0.54216665, 0.9560631, 0.9991162, 0.0],
#                 [0.7506478, 0.08314464, 0.2933517, 0.8858382, 0.0],
#                 [0.9961341, 0.8858382, 0.2933517, 0.08314464, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0],
#             ],
#             [
#                 [0.0, 0.7506478, 0.9961341, 0.0, 0.0],
#                 [0.9560631, 0.2933517, 0.2933517, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0],
#             ],
#             [
#                 [0.0, 0.39346933, 0.86466473, 0.988891, 0.99966455],
#                 [0.2933517, 0.01379288, 0.49366438, 0.90436554, 0.993355],
#                 [0.7506478, 0.1992626, 0.05404053, 0.5888877, 0.93427145],
#                 [0.9560631, 0.6753475, 0.1175031, 0.1175031, 0.6753475],
#                 [0.9961341, 0.93427145, 0.5888877, 0.05404053, 0.1992626],
#                 [0.9998301, 0.993355, 0.90436554, 0.49366438, 0.01379288],
#             ],
#         ],
#         ,
#     )
#     assert paddle.allclose(soft_mask, ref_soft_mask)


# def test_guided_attention_loss_value(device):
#     paddle.device.set_device(device)
#     from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss

#     loss = GuidedAttentionLoss().to(device)
#     input_lengths = paddle.tensor([2, 3], )
#     target_lengths = paddle.tensor([3, 4], )
#     alignments = paddle.tensor(
#         [
#             [
#                 [0.8, 0.2, 0.0],
#                 [0.4, 0.6, 0.0],
#                 [0.2, 0.8, 0.0],
#                 [0.0, 0.0, 0.0],
#             ],
#             [
#                 [0.6, 0.2, 0.2],
#                 [0.1, 0.7, 0.2],
#                 [0.3, 0.4, 0.3],
#                 [0.2, 0.3, 0.5],
#             ],
#         ],
#         ,
#     )
#     loss_value = loss(alignments, input_lengths, target_lengths)
#     ref_loss_value = paddle.tensor(0.1142)
#     assert paddle.isclose(loss_value, ref_loss_value, 0.0001, 0.0001).item()


# def test_guided_attention_loss_shapes(device):
#     paddle.device.set_device(device)
#     from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss

#     loss = GuidedAttentionLoss().to(device)
#     input_lengths = paddle.tensor([3, 2, 6], )
#     output_lengths = paddle.tensor([4, 3, 5], )
#     soft_mask = loss.guided_attentions(input_lengths, output_lengths)
#     assert soft_mask.shape == (3, 6, 5)
#     soft_mask = loss.guided_attentions(
#         input_lengths, output_lengths, max_input_len=10
#     )
#     assert soft_mask.shape == (3, 10, 5)
#     soft_mask = loss.guided_attentions(
#         input_lengths, output_lengths, max_target_len=12
#     )
#     assert soft_mask.shape == (3, 6, 12)
#     soft_mask = loss.guided_attentions(
#         input_lengths, output_lengths, max_input_len=10, max_target_len=12
#     )
#     assert soft_mask.shape == (3, 10, 12)
