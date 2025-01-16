# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/ml/test_decorators.py)
import sys
import time

import paddle
from visualdl import LogWriter

from paddlespeech.audiotools import util
from paddlespeech.audiotools.ml.decorators import timer
from paddlespeech.audiotools.ml.decorators import Tracker
from paddlespeech.audiotools.ml.decorators import when


def test_all_decorators():
    rank = 0
    max_iters = 100

    writer = LogWriter("/tmp/logs")
    tracker = Tracker(writer, log_file="/tmp/log.txt")

    train_data = range(100)
    val_data = range(100)

    @tracker.log("train", "value", history=False)
    @tracker.track("train", max_iters, tracker.step)
    @timer()
    def train_loop():
        i = tracker.step
        time.sleep(0.01)
        return {
            "loss":
            util.exp_compat(paddle.to_tensor([-i / 100], dtype="float32")),
            "mel":
            util.exp_compat(paddle.to_tensor([-i / 100], dtype="float32")),
            "stft":
            util.exp_compat(paddle.to_tensor([-i / 100], dtype="float32")),
            "waveform":
            util.exp_compat(paddle.to_tensor([-i / 100], dtype="float32")),
            "not_scalar":
            paddle.arange(start=0, end=10, step=1, dtype="int64"),
        }

    @tracker.track("val", len(val_data))
    @timer()
    def val_loop():
        i = tracker.step
        time.sleep(0.01)
        return {
            "loss":
            util.exp_compat(paddle.to_tensor([-i / 100], dtype="float32")),
            "mel":
            util.exp_compat(paddle.to_tensor([-i / 100], dtype="float32")),
            "stft":
            util.exp_compat(paddle.to_tensor([-i / 100], dtype="float32")),
            "waveform":
            util.exp_compat(paddle.to_tensor([-i / 100], dtype="float32")),
            "not_scalar":
            paddle.arange(10, dtype="int64"),
            "string":
            "string",
        }

    @when(lambda: tracker.step % 1000 == 0 and rank == 0)
    @paddle.no_grad()
    def save_samples():
        tracker.print("Saving samples to TensorBoard.")

    @when(lambda: tracker.step % 100 == 0 and rank == 0)
    def checkpoint():
        save_samples()
        if tracker.is_best("val", "mel"):
            tracker.print("Best model so far.")
        tracker.print("Saving to /runs/exp1")
        tracker.done("val", f"Iteration {tracker.step}")

    @when(lambda: tracker.step % 100 == 0)
    @tracker.log("val", "mean")
    @paddle.no_grad()
    def validate():
        for _ in range(len(val_data)):
            output = val_loop()
        return output

    with tracker.live:
        for tracker.step in range(max_iters):
            validate()
            checkpoint()
            train_loop()

    state_dict = tracker.state_dict()
    tracker.load_state_dict(state_dict)

    # If train loop returned not a dict
    @tracker.track("train", max_iters, tracker.step)
    def train_loop_2():
        i = tracker.step
        time.sleep(0.01)

    with tracker.live:
        for tracker.step in range(max_iters):
            validate()
            checkpoint()
            train_loop_2()


if __name__ == "__main__":
    test_all_decorators()
