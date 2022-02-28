import pytest
import paddle

def test_checkpointer(tmpdir, device):
    paddle.device.set_device(device)
    from speechbrain.utils.checkpoints import Checkpointer    
    class Recoverable(paddle.nn.Layer):
        def __init__(self, param):
            super().__init__()
            param = paddle.to_tensor([param], dtype="float32")
            
            self.param = paddle.create_parameter(param.shape, dtype=param.dtype)
            self.param.set_value(param)
            # print("param shape: {}".format(self.param.shape))
        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(2.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)
    recoverable.param.set_value(paddle.to_tensor([1.0], ))
    # Should not be possible since no checkpoint saved yet:
    assert not recoverer.recover_if_possible()
    result = recoverable(10.0)
    # Check that parameter has not been loaded from original value:
    assert recoverable.param.equal(paddle.to_tensor([1.0], ))

    ckpt = recoverer.save_checkpoint()
    # Check that the name recoverable has a save file:
    # NOTE: Here assuming .pt filename; if convention changes, change test
    assert (ckpt.path / "recoverable.ckpt").exists()
    # Check that saved checkpoint is found, and location correct:
    assert recoverer.list_checkpoints()[0] == ckpt
    assert recoverer.list_checkpoints()[0].path.parent == tmpdir
    recoverable.param.set_value(paddle.to_tensor([2.0], ))
    recoverer.recover_if_possible()
    # Check that parameter has been loaded immediately:
    
    assert recoverable.param.equal(paddle.to_tensor([1.0], ))
    result = recoverable(10.0)
    # And result correct
    assert result == 10.0

    other = Recoverable(2.0)
    recoverer.add_recoverable("other", other)
    # Check that both objects are now found:
    assert recoverer.recoverables["recoverable"] == recoverable
    assert recoverer.recoverables["other"] == other
    new_ckpt = recoverer.save_checkpoint()
    # Check that now both recoverables have a save file:
    assert (new_ckpt.path / "recoverable.ckpt").exists()
    assert (new_ckpt.path / "other.ckpt").exists()
    assert new_ckpt in recoverer.list_checkpoints()
    recoverable.param.set_value(paddle.to_tensor([2.0], ))
    other.param.set_value(paddle.to_tensor([10.0], ))
    chosen_ckpt = recoverer.recover_if_possible()
    # Should choose newest by default:
    assert chosen_ckpt == new_ckpt
    # Check again that parameters have been loaded immediately:
    assert recoverable.param.equal(paddle.to_tensor([1.0], ))
    assert other.param.equal(paddle.to_tensor([2.0], ))
    other_result = other(10.0)
    # And again we should have the correct computations:
    assert other_result == 20.0

    # Recover from oldest, which does not have "other":
    # This also tests a custom sort
    # Raises by default:
    with pytest.raises(RuntimeError):
        chosen_ckpt = recoverer.recover_if_possible(
            importance_key=lambda x: -x.meta["unixtime"]
        )
    # However this operation may have loaded the first object
    # so let's set the values manually:
    recoverable.param.set_value(paddle.to_tensor([2.0], ))
    other.param.set_value(paddle.to_tensor([10.0], ))
    recoverer.allow_partial_load = True
    chosen_ckpt = recoverer.recover_if_possible(
        importance_key=lambda x: -x.meta["unixtime"]
    )
    # Should have chosen the original:
    assert chosen_ckpt == ckpt
    # And should recover recoverable:
    assert recoverable.param.equal(paddle.to_tensor([1.0], ))
    # But not other:
    other_result = other(10.0)
    assert other.param.equal(paddle.to_tensor([10.0], ))
    assert other_result == 100.0

    # Test saving names checkpoints with meta info, and custom filter
    epoch_ckpt = recoverer.save_checkpoint(name="ep1", meta={"loss": 2.0})
    assert "ep1" in epoch_ckpt.path.name
    other.param.set_value(paddle.to_tensor([2.0], ))
    recoverer.save_checkpoint(meta={"loss": 3.0})
    chosen_ckpt = recoverer.recover_if_possible(
        ckpt_predicate=lambda ckpt: "loss" in ckpt.meta,
        importance_key=lambda ckpt: -ckpt.meta["loss"],
    )
    assert chosen_ckpt == epoch_ckpt
    assert other.param.equal(paddle.to_tensor([10.0], ))

    # Make sure checkpoints can't be name saved by the same name
    with pytest.raises(FileExistsError):
        recoverer.save_checkpoint(name="ep1")


def test_recovery_custom_io(tmpdir):
    from speechbrain.utils.checkpoints import register_checkpoint_hooks
    from speechbrain.utils.checkpoints import mark_as_saver
    from speechbrain.utils.checkpoints import mark_as_loader
    from speechbrain.utils.checkpoints import Checkpointer

    @register_checkpoint_hooks
    class CustomRecoverable:
        def __init__(self, param):
            self.param = int(param)

        @mark_as_saver
        def save(self, path):
            with open(path, "w") as fo:
                fo.write(str(self.param))

        @mark_as_loader
        def load(self, path, end_of_epoch, device):
            del end_of_epoch  # Unused
            del device
            with open(path) as fi:
                self.param = int(fi.read())

    custom_recoverable = CustomRecoverable(0)
    recoverer = Checkpointer(tmpdir, {"custom_recoverable": custom_recoverable})
    custom_recoverable.param = 1
    # First, make sure no checkpoints are found
    # (e.g. somehow tmpdir contaminated)
    ckpt = recoverer.recover_if_possible()
    assert ckpt is None
    ckpt = recoverer.save_checkpoint()
    custom_recoverable.param = 2
    loaded_ckpt = recoverer.recover_if_possible()
    # Make sure we got the same thing:
    assert ckpt == loaded_ckpt
    # With this custom recoverable, the load is instant:
    assert custom_recoverable.param == 1


def test_checkpoint_deletion(tmpdir, device):
    import paddle
    from speechbrain.utils.checkpoints import Checkpointer

    class Recoverable(paddle.nn.Layer):
        def __init__(self, param):
            super().__init__()
            param = paddle.to_tensor([param], dtype="float32")
            self.param = paddle.create_parameter(
                param.shape, dtype="float32")
            self.param.set_value(param)
        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(1.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)
    first_ckpt = recoverer.save_checkpoint()
    recoverer.delete_checkpoints()
    # Will not delete only checkpoint by default:
    assert first_ckpt in recoverer.list_checkpoints()
    second_ckpt = recoverer.save_checkpoint()
    recoverer.delete_checkpoints()
    # Oldest checkpoint is deleted by default:
    assert first_ckpt not in recoverer.list_checkpoints()
    # Other syntax also should work:
    recoverer.save_and_keep_only()
    assert second_ckpt not in recoverer.list_checkpoints()
    # Can delete all checkpoints:
    recoverer.delete_checkpoints(num_to_keep=0)
    assert not recoverer.list_checkpoints()

    # Now each should be kept:
    # Highest foo
    c1 = recoverer.save_checkpoint(meta={"foo": 2})
    # Latest CKPT after filtering
    c2 = recoverer.save_checkpoint(meta={"foo": 1})
    # Filtered out
    c3 = recoverer.save_checkpoint(meta={"epoch_ckpt": True})
    recoverer.delete_checkpoints(
        num_to_keep=1,
        max_keys=["foo"],
        importance_keys=[lambda c: c.meta["unixtime"]],
        ckpt_predicate=lambda c: "epoch_ckpt" not in c.meta,
    )
    assert all(c in recoverer.list_checkpoints() for c in [c1, c2, c3])
    # Reset:
    recoverer.delete_checkpoints(num_to_keep=0)
    assert not recoverer.list_checkpoints()

    # Test the keeping multiple checkpoints without predicate:
    # This should be deleted:
    c_to_delete = recoverer.save_checkpoint(meta={"foo": 2})
    # Highest foo
    c1 = recoverer.save_checkpoint(meta={"foo": 3})
    # Latest CKPT after filtering
    c2 = recoverer.save_checkpoint(meta={"foo": 1})
    recoverer.delete_checkpoints(
        num_to_keep=1,
        importance_keys=[lambda c: c.meta["unixtime"], lambda c: c.meta["foo"]],
    )
    assert all(c in recoverer.list_checkpoints() for c in [c1, c2])
    assert c_to_delete not in recoverer.list_checkpoints()


def test_multiple_ckpts_and_criteria(tmpdir):
    from speechbrain.utils.checkpoints import Checkpointer
    import paddle

    class Recoverable(paddle.nn.Layer):
        def __init__(self, param):
            super().__init__()
            param = paddle.to_tensor([param], dtype="float32")
            self.param = paddle.create_parameter(
                param.shape, dtype="float32")
            self.param.set_value(param)

        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(1.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)

    # Here testing multiple checkpoints with equal meta criteria
    recoverer.save_and_keep_only(
        meta={"error": 5}, min_keys=["error"], keep_recent=True
    )
    # By default, get the most recent one:
    first_ckpt = recoverer.find_checkpoint()
    recoverer.save_and_keep_only(
        meta={"error": 5}, min_keys=["error"], keep_recent=True
    )
    second_ckpt = recoverer.find_checkpoint()
    assert first_ckpt.meta["unixtime"] < second_ckpt.meta["unixtime"]
    recoverer.save_and_keep_only(
        meta={"error": 6}, min_keys=["error"], keep_recent=True
    )
    third_ckpt = recoverer.find_checkpoint()
    remaining_ckpts = recoverer.list_checkpoints()
    assert first_ckpt not in remaining_ckpts
    assert second_ckpt in remaining_ckpts
    assert third_ckpt in remaining_ckpts

    # With equal importance criteria, the latest checkpoint should always be
    # returned
    fourth_ckpt = recoverer.save_checkpoint(meta={"error": 5})
    found_ckpt = recoverer.find_checkpoint(min_key="error")
    assert found_ckpt == fourth_ckpt
    fifth_ckpt = recoverer.save_checkpoint(meta={"error": 5})
    # Similarly for getting multiple checkpoints:
    found_ckpts = recoverer.find_checkpoints(
        min_key="error", max_num_checkpoints=2
    )
    assert found_ckpts == [fifth_ckpt, fourth_ckpt]


def test_torch_meta(tmpdir, device):
    from speechbrain.utils.checkpoints import Checkpointer
    import paddle

    class Recoverable(paddle.nn.Layer):
        def __init__(self, param):
            super().__init__()
            param = paddle.to_tensor([param], dtype="float32")
            self.param = paddle.create_parameter(
                param.shape, dtype="float32")
            self.param.set_value(param)

        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(1.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)
    saved = recoverer.save_checkpoint(
        meta={"loss": paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0], ).numpy()}
    )
    loaded = recoverer.recover_if_possible()

    # assert 0
    assert paddle.to_tensor(saved.meta["loss"]).allclose(paddle.to_tensor(loaded.meta["loss"]))


def test_checkpoint_hook_register(tmpdir):
    from speechbrain.utils.checkpoints import register_checkpoint_hooks
    from speechbrain.utils.checkpoints import mark_as_saver
    from speechbrain.utils.checkpoints import mark_as_loader
    from speechbrain.utils.checkpoints import Checkpointer

    # First a proper interface:
    @register_checkpoint_hooks
    class CustomRecoverable:
        def __init__(self, param):
            self.param = int(param)

        @mark_as_saver
        def save(self, path):
            with open(path, "w") as fo:
                fo.write(str(self.param))

        @mark_as_loader
        def load(self, path, end_of_epoch, device):
            del end_of_epoch  # Unused
            with open(path) as fi:
                self.param = int(fi.read())

    recoverable = CustomRecoverable(1.0)
    checkpointer = Checkpointer(tmpdir, {"recoverable": recoverable})
    checkpointer.save_checkpoint()
    recoverable.param = 2.0
    checkpointer.recover_if_possible()
    assert recoverable.param == 1.0

    # Improper interfaces:
    with pytest.raises(TypeError):

        class BadRecoverable:
            def __init__(self, param):
                self.param = int(param)

            def save(self, path):
                with open(path, "w") as fo:
                    fo.write(str(self.param))

            @mark_as_loader
            def load(self, path, end_of_epoch):  # MISSING device
                del end_of_epoch  # Unused
                with open(path) as fi:
                    self.param = int(fi.read())

    with pytest.raises(TypeError):

        class BadRecoverable:  # noqa: F811
            def __init__(self, param):
                self.param = int(param)

            @mark_as_saver
            def save(self, path, extra_arg):  # Extra argument
                with open(path, "w") as fo:
                    fo.write(str(self.param))

            def load(self, path, end_of_epoch, device):
                del end_of_epoch  # Unused
                with open(path) as fi:
                    self.param = int(fi.read())


# 暂时先不验证这个部分
# 由于部分LR在Paddle中未实现，因此现在不验证
# def test_paddle_defaults(tmpdir, device):
#     paddle.device.set_device(device)
#     from speechbrain.utils.checkpoints import Checkpointer

#     module = paddle.nn.Linear(10, 10)
#     optimizer = paddle.optimizer.Adam(parameters=module.parameters())
#     lr_scheduler = paddle.optimizer.lr.CyclicLR(
#         optimizer, 0.1, 1.0, cycle_momentum=False
#     )
#     # ReduceLROnPlateau is on an _LRScheduler for some reason, so have a separate test for it
#     another_scheduler = paddle.optimizer.lr_scheduler.ReduceLROnPlateau(optimizer)
#     checkpointer = Checkpointer(
#         tmpdir,
#         recoverables={
#             "module": module,
#             "optimizer": optimizer,
#             "scheduler": lr_scheduler,
#             "scheduler2": another_scheduler,
#         },
#     )
#     ckpt = checkpointer.save_checkpoint()
#     # test the module:
#     inp = paddle.randn((3, 10), )
#     prev_output = module(inp)

#     # Re-initialize everything
#     module = paddle.nn.Linear(10, 10, )
#     optimizer = paddle.optim.Adam(module.parameters())
#     lr_scheduler = paddle.optim.lr_scheduler.CyclicLR(
#         optimizer, 0.1, 1.0, cycle_momentum=False
#     )
#     another_scheduler = paddle.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#     checkpointer = Checkpointer(
#         tmpdir,
#         recoverables={
#             "module": module,
#             "optimizer": optimizer,
#             "scheduler": lr_scheduler,
#             "scheduler2": another_scheduler,
#         },
#     )
#     checkpointer.load_checkpoint(ckpt)
#     assert paddle.allclose(module(inp), prev_output)
