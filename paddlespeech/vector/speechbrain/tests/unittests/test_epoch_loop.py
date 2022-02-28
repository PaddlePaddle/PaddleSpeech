def test_epoch_loop_recovery(tmpdir):
    from speechbrain.utils.checkpoints import Checkpointer
    from speechbrain.utils.epoch_loop import EpochCounter

    epoch_counter = EpochCounter(2)
    recoverer = Checkpointer(tmpdir, {"epoch": epoch_counter})
    for epoch in epoch_counter:
        assert epoch == 1
        # Save a mid-epoch checkpoint:
        recoverer.save_checkpoint(end_of_epoch=False)
        # Simulate interruption
        break
    # Now after recovery still at epoch 1:
    recoverer.recover_if_possible()
    second_epoch = False  # Will manually update this
    for epoch in epoch_counter:
        if not second_epoch:
            assert epoch == 1
            recoverer.save_checkpoint(end_of_epoch=True)
            second_epoch = True
        else:
            assert epoch == 2
            # Again simulate interruption
            break
    # Now after recovery we are in epoch 2:
    recoverer.recover_if_possible()
    loop_runs = 0
    for epoch in epoch_counter:
        assert epoch == 2
        loop_runs += 1
        recoverer.save_checkpoint(end_of_epoch=True)
    # And that is that:
    assert loop_runs == 1
    # And now after recovery, no more epochs:
    recoverer.recover_if_possible()
    for epoch in epoch_counter:
        # Will not get here:
        assert False
