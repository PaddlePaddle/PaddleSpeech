import paddle

def test_parse_arguments():
    paddle.device.set_device("cpu")
    from speechbrain.core import parse_arguments

    filename, run_opts, overrides = parse_arguments(
        ["params.yaml", "--device=cpu", "--seed=3", "--data_folder", "TIMIT"]
    )
    assert filename == "params.yaml"
    assert run_opts["device"] == "cpu"
    assert overrides == "seed: 3\ndata_folder: TIMIT"


def test_brain(device):
    import paddle
    from speechbrain.core import Brain, Stage
    from paddle.optimizer import SGD

    model = paddle.nn.Linear(in_features=10, out_features=10)

    class SimpleBrain(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return paddle.nn.functional.l1_loss(predictions, batch[1])

    # Paddle 中 SGD 参数和 learning_rate的位置对换一下
    brain = SimpleBrain(
        {"model": model}, lambda x, grad_clip=None: SGD(0.1, x, grad_clip=grad_clip), run_opts={"device": device}
    )

    inputs = paddle.rand([10, 10])
    targets = paddle.rand([10, 10])
    train_set = ([inputs, targets],)
    valid_set = ([inputs, targets],)

    start_output = brain.compute_forward(inputs, Stage.VALID)
    start_loss = brain.compute_objectives(start_output, targets, Stage.VALID)

    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)
    end_output = brain.compute_forward(inputs, Stage.VALID)
    end_loss = brain.compute_objectives(end_output, targets, Stage.VALID)
    assert end_loss < start_loss
