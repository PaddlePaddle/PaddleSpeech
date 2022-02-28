def test_pretrainer(tmpdir, device):
    import paddle
    import os
    from paddle.nn import Linear
    paddle.device.set_device("cpu")

    # save a model in tmpdir/original/model.ckpt
    first_model = Linear(32, 32)
    tmpdir = "./"
    pretrained_dir = os.path.join(tmpdir, "original")
    # pretrained_dir = tmpdir / "original"
    # print("type: {}".format(type(pretrained_dir)))
    # pretrained_dir.mkdir()
    if not os.path.exists(pretrained_dir):
        os.mkdir(pretrained_dir)
    model_dir = os.path.join(pretrained_dir,  "model.pdparams")
    paddle.save(first_model.state_dict(), model_dir)

    # Make a new model and Pretrainer
    pretrained_model = Linear(32, 32)
    assert not paddle.all(paddle.equal(pretrained_model.weight, first_model.weight))
    from speechbrain.utils.parameter_transfer import Pretrainer

    pt = Pretrainer(
        collect_in=os.path.join(tmpdir, "reused"), loadables={"model": pretrained_model}
    )
    pt.collect_files(default_source=pretrained_dir)
    pt.load_collected()
    assert paddle.all(paddle.equal(pretrained_model.weight, first_model.weight))
