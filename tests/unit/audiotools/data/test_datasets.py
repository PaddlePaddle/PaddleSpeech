# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/data/test_datasets.py)
import sys
import tempfile
from pathlib import Path

import numpy as np
import paddle
import pytest

from paddlespeech import audiotools
from paddlespeech.audiotools.data import transforms as tfm


def test_align_lists():
    input_lists = [
        ["a/1.wav", "b/1.wav", "c/1.wav", "d/1.wav"],
        ["a/2.wav", "c/2.wav"],
        ["c/3.wav"],
    ]
    target_lists = [
        ["a/1.wav", "b/1.wav", "c/1.wav", "d/1.wav"],
        ["a/2.wav", "none", "c/2.wav", "none"],
        ["none", "none", "c/3.wav", "none"],
    ]

    def _preprocess(lists):
        output = []
        for x in lists:
            output.append([])
            for y in x:
                output[-1].append({"path": y})
        return output

    input_lists = _preprocess(input_lists)
    target_lists = _preprocess(target_lists)

    aligned_lists = audiotools.datasets.align_lists(input_lists)
    assert target_lists == aligned_lists


def test_audio_dataset():
    transform = tfm.Compose(
        [
            tfm.VolumeNorm(),
            tfm.Silence(prob=0.5),
        ], )
    loader = audiotools.data.datasets.AudioLoader(
        sources=["./audio/spk.csv"],
        transform=transform, )
    dataset = audiotools.data.datasets.AudioDataset(
        loader,
        44100,
        n_examples=100,
        transform=transform, )
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=dataset.collate, )
    for batch in dataloader:
        kwargs = batch["transform_args"]
        signal = batch["signal"]
        original = signal.clone()

        signal = dataset.transform(signal, **kwargs)
        original = dataset.transform(original, **kwargs)

        mask = kwargs["Compose"]["1.Silence"]["mask"]

        zeros_ = paddle.zeros_like(signal[mask].audio_data)
        original_ = original[~mask].audio_data

        assert paddle.allclose(signal[mask].audio_data, zeros_)
        assert paddle.allclose(signal[~mask].audio_data, original_)


def test_aligned_audio_dataset():
    with tempfile.TemporaryDirectory() as d:
        dataset_dir = Path(d)
        audiotools.util.generate_chord_dataset(
            max_voices=8, num_items=3, output_dir=dataset_dir)
        loaders = [
            audiotools.data.datasets.AudioLoader([dataset_dir / f"track_{i}"])
            for i in range(3)
        ]
        dataset = audiotools.data.datasets.AudioDataset(
            loaders, 44100, n_examples=1000, aligned=True, shuffle_loaders=True)
        dataloader = paddle.io.DataLoader(
            dataset,
            batch_size=16,
            num_workers=0,
            collate_fn=dataset.collate, )

        # Make sure the voice tracks are aligned.
        for batch in dataloader:
            paths = []
            for i in range(len(loaders)):
                _paths = [p.split("/")[-1] for p in batch[i]["path"]]
                paths.append(_paths)
            paths = np.array(paths)
            for i in range(paths.shape[1]):
                col = paths[:, i]
                col = col[col != "none"]
                assert np.all(col == col[0])


def test_loader_without_replacement():
    with tempfile.TemporaryDirectory() as d:
        dataset_dir = Path(d)
        num_items = 100
        audiotools.util.generate_chord_dataset(
            max_voices=1,
            num_items=num_items,
            output_dir=dataset_dir,
            duration=0.01, )
        loader = audiotools.data.datasets.AudioLoader(
            [dataset_dir], shuffle=False)
        dataset = audiotools.data.datasets.AudioDataset(loader, 44100)

        for idx in range(num_items):
            item = dataset[idx]
            assert item["item_idx"] == idx


def test_loader_with_replacement():
    with tempfile.TemporaryDirectory() as d:
        dataset_dir = Path(d)
        num_items = 100
        audiotools.util.generate_chord_dataset(
            max_voices=1,
            num_items=num_items,
            output_dir=dataset_dir,
            duration=0.01, )
        loader = audiotools.data.datasets.AudioLoader([dataset_dir])
        dataset = audiotools.data.datasets.AudioDataset(
            loader, 44100, without_replacement=False)

        for idx in range(num_items):
            item = dataset[idx]


def test_loader_out_of_range():
    with tempfile.TemporaryDirectory() as d:
        dataset_dir = Path(d)
        num_items = 100
        audiotools.util.generate_chord_dataset(
            max_voices=1,
            num_items=num_items,
            output_dir=dataset_dir,
            duration=0.01, )
        loader = audiotools.data.datasets.AudioLoader([dataset_dir])

        item = loader(
            sample_rate=44100,
            duration=0.01,
            state=audiotools.util.random_state(0),
            source_idx=0,
            item_idx=101, )
        assert item["path"] == "none"


def test_dataset_pipeline():
    transform = tfm.Compose([
        tfm.RoomImpulseResponse(sources=["./audio/irs.csv"]),
        tfm.BackgroundNoise(sources=["./audio/noises.csv"]),
    ])
    loader = audiotools.data.datasets.AudioLoader(sources=["./audio/spk.csv"])
    dataset = audiotools.data.datasets.AudioDataset(
        loader,
        44100,
        n_examples=10,
        transform=transform, )
    dataloader = paddle.io.DataLoader(
        dataset, num_workers=0, batch_size=1, collate_fn=dataset.collate)
    for batch in dataloader:
        batch = audiotools.core.util.prepare_batch(batch, device="cpu")
        kwargs = batch["transform_args"]
        signal = batch["signal"]
        batch = dataset.transform(signal, **kwargs)


class NumberDataset:
    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {"idx": idx}


def test_concat_dataset():
    d1 = NumberDataset()
    d2 = NumberDataset()
    d3 = NumberDataset()

    d = audiotools.datasets.ConcatDataset([d1, d2, d3])
    x = d.collate([d[i] for i in range(len(d))])["idx"].tolist()

    t = []
    for i in range(10):
        t += [i, i, i]

    assert x == t
