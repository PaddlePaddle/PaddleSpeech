import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import argbind
import paddle
import paddle.nn as nn
from visualdl import LogWriter

import paddlespeech
import paddlespeech.t2s.modules.losses as _losses
from paddlespeech.audiotools.core import AudioSignal
from paddlespeech.audiotools.core import util
from paddlespeech.audiotools.data import transforms
from paddlespeech.audiotools.data.datasets import AudioDataset
from paddlespeech.audiotools.data.datasets import AudioLoader
from paddlespeech.audiotools.data.datasets import ConcatDataset
from paddlespeech.audiotools.ml import Accelerator
from paddlespeech.audiotools.ml.decorators import timer
from paddlespeech.audiotools.ml.decorators import Tracker
from paddlespeech.audiotools.ml.decorators import when
from paddlespeech.codec.models.dac_.model import DAC
from paddlespeech.codec.models.dac_.model import Discriminator
from paddlespeech.t2s.training.seeding import seed_everything

warnings.filterwarnings("ignore", category=UserWarning)

# Optimizers
AdamW = argbind.bind(paddle.optimizer.AdamW, "generator", "discriminator")
# Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float=1.0):
    return paddle.optimizer.lr.ExponentialDecay(optimizer, gamma)


# Models
# DAC = argbind.bind(dac.model.DAC)
# Discriminator = argbind.bind(dac.model.Discriminator)
DAC = argbind.bind(DAC)
Discriminator = argbind.bind(Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")


# Loss
# filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
def filter_fn(fn):
    return hasattr(fn, "forward") and "Loss" in fn.__name__


losses = argbind.bind_module(_losses, filter_fn=filter_fn)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


# Transforms
# filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
#     "BaseTransform",
#     "Compose",
#     "Choose", ]
def filter_fn(fn):
    return hasattr(fn, "transform") and fn.__qualname__ not in [
        "BaseTransform",
        "Compose",
        "Choose",
    ]


tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)


# to_tfm = lambda l: [getattr(tfm, x)() for x in l]
def to_tfm(l):
    return [getattr(tfm, x)() for x in l]


@argbind.bind("train", "val")
def build_transform(
        augment_prob: float=1.0,
        preprocess: list=["Identity"],
        augment: list=["Identity"],
        postprocess: list=["Identity"], ):
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(
        *to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
        sample_rate: int,
        folders: dict=None, ):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: nn.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker


# @argbind.bind(without_prefix=True)
def load(
        args,
        accel: Accelerator,
        tracker: Tracker,
        save_path: str,
        resume: bool=False,
        tag: str="latest",
        load_weights: bool=False, ):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        tracker.print(
            f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac").exists():
            generator, g_extra = DAC.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = DAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    waveform_loss = nn.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data, )


@timer()
@paddle.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(batch["signal"].clone(),
                                      **batch["transform_args"])

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }


@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    with paddle.no_grad():
        signal = state.train_data.transform(batch["signal"].clone(),
                                            **batch["transform_args"])

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]

    with accel.autocast():
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons,
                                                                    signal)

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = paddle.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0)
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        (output["adv/gen_loss"],
         output["adv/feat_loss"], ) = state.gan_loss.generator_loss(recons,
                                                                    signal)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss
        output["loss"] = sum(
            [v * output[k] for k, v in lambdas.items() if k in output])

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = paddle.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3)
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print("Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(f"{save_path}/{tag}",
                                                     generator_extra)
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(f"{save_path}/{tag}",
                                                         discriminator_extra)


@paddle.no_grad()
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.train_data.transform(batch["signal"].clone(),
                                        **batch["transform_args"])

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(f"{k}/sample_{nb}.wav", writer,
                                          state.tracker.step)


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


# @argbind.bind(without_prefix=True)
def train(
        args,
        accel: Accelerator,
        seed: int=2025,
        save_path: str="ckpt",
        num_iters: int=250000,
        save_iters: list=[10000, 50000, 100000, 200000],
        sample_freq: int=10000,
        valid_freq: int=1000,
        batch_size: int=12,
        val_batch_size: int=10,
        num_workers: int=8,
        val_idx: list=[0, 1, 2, 3, 4, 5, 6, 7],
        lambdas: dict={
            "mel/loss": 100.0,
            "adv/feat_loss": 2.0,
            "adv/gen_loss": 1.0,
            "vq/commitment_loss": 0.25,
            "vq/codebook_loss": 1.0,
        }, ):
    seed_everything(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = LogWriter(
        log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank)

    state = load(args, accel, tracker, save_path)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate, )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False, )

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log(
        "train", "value", history=False)(tracker.track(
            "train", num_iters, completed=state.tracker.step)(train_loop))
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(
                train_dataloader, start=tracker.step):
            train_loop(state, batch, accel, lambdas)

            last_iter = (tracker.step == num_iters - 1
                         if num_iters is not None else False)
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, writer)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    # args = argbind.parse_args()
    # args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    # with argbind.scope(args):
    with Accelerator() as accel:
        if accel.local_rank != 0:
            sys.tracebacklimit = 0
        # train(args, accel)
        train(None, accel)
