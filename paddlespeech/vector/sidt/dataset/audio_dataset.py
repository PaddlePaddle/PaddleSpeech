#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2021    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
Load audio dataset
"""

import sys
import random
import numpy as np

import sidt.utils.utils as utils
from sidt import _logger as log
from sidt.utils.data_utils import batch_pad_right
from sidt.utils.utils import read_map_file
from sidt.utils.data_utils import pad_right_to
import sidt.utils.features as feats
from sidt.dataset.augment import Tempo, AddNoise, ChangeVolume, Clicks, Clip, Reverb
import traceback


def get_audio_collate_fn():
    """
    A wrapper for generate collate function.

    Returns:
        audio_collate_fn: collate funtion for dataloader
    """
    def audio_collate_fn(batch):
        """
        Custom collate function for audio dataset

        Returns:
            audio_collate_fn: collate funtion for dataloader
        """

        data = []
        target = []
        for items in batch:
            for x, y in zip(items[0], items[1]):
                data.append(np.array(x))
                target.append(y)

        data, lengths = batch_pad_right(data)
        return np.array(data, dtype=np.float32), \
               np.array(lengths, dtype=np.float32), \
               np.array(target, dtype=np.long).reshape((len(target), 1))

    return audio_collate_fn


def create_audio_dataset(cls):
    """audio dataset factory

    Args:
        cls: base cls of dataset, paddle dataset or pytorch dataset

    Returns:
        datset: an dataset instance of input class
    """

    class AudioDataset(cls):
        """
        Dataset used to load audio files
        """
        def __init__(self, scp_file, label2utt, min_item_size=1,
                     max_item_size=1, repeat=1, min_chunk_size=3,
                     max_chunk_size=10, select_by_speaker=True,
                     sample_rate=8000, augment_pipelines=[],
                     spec_aug_pipelines=[], feat_type="fbank",
                     num_mel_bins=40, num_ceps=40):
            self.scp_file = scp_file
            self.scp_reader = None
            self.repeat = repeat
            self.min_item_size = min_item_size
            self.max_item_size = max_item_size
            self.min_chunk_size = min_chunk_size
            self.max_chunk_size = max_chunk_size
            self._collate_fn = get_audio_collate_fn()
            self._is_select_by_speaker = select_by_speaker

            utt2wav, wav2utt = read_map_file(scp_file, values_func=lambda x:x[0])
            self.utt2wav = utt2wav

            label2utts, utt2label = read_map_file(label2utt, key_func=int)
            self.utt_info = list(label2utts.items()) if self._is_select_by_speaker else list(utt2label.items())

            self.sample_rate = sample_rate
            self.augment_pipelines = augment_pipelines
            self.spec_aug_pipelines = spec_aug_pipelines

            assert feat_type in ["fbank", "mfcc"]
            self.feat_type = feat_type
            self.num_mel_bins = num_mel_bins
            self.num_ceps = num_ceps

        @property
        def collate_fn(self):
            """
            Return a collate funtion.
            """
            return self._collate_fn

        def _select_by_speaker(self, index):
            if not self.utt2wav or not self.utt_info:
                return []
            index = index % (len(self.utt_info))
            inputs = []
            labels = []
            label = self.utt_info[index][0]
            utts = self.utt_info[index][1]
            item_size = random.randint(self.min_item_size, self.max_item_size)
            for loop_idx in range(item_size):
                try:
                    utt_index = random.randint(0, len(utts)) % len(utts)
                    utt = utts[utt_index]
                except:
                    print(index, utt_index, len(self.utt_info[index][1]))
                    raise

                x, y = self._compute_feats(utt, label)
                inputs.extend(x)
                labels.extend(y)
            return inputs, labels

        def _select_by_utt(self, index):
            if not self.utt2wav or not self.utt_info:
                return []
            index = index % (len(self.utt_info))
            utt = self.utt_info[index][0]

            x, y = self._compute_feats(utt, self.utt_info[index][1])

            return x, y

        def _compute_feats(self, utt, label):
            try:
                chunk_size = random.random() * (self.max_chunk_size - self.min_chunk_size) + self.min_chunk_size
                sig = feats.load_audio(self.utt2wav[utt], duration=chunk_size,
                                       target_sample_rate=self.sample_rate)
                sigs = [sig]
                for aug_pipeline in self.augment_pipelines:
                    aug_sig = aug_pipeline(sig)
                    if aug_sig.ndim != sig.ndim:
                        aug_sig = np.expand_dims(aug_sig, axis=0)
                    sigs.append(aug_sig)

                all_feats = []
                for sig in sigs:
                    feat = feats.fbank(sig, self.num_mel_bins, self.sample_rate) if self.feat_type == "fbank" else\
                            feats.mfcc(sig, self.num_mel_bins, self.num_ceps, self.sample_rate)
                    feat = feats.sliding_window_cmvn(feat)
                    feat = np.transpose(feat)
                    vad_feat = feats.vad(feat)
                    if vad_feat.shape[1] < 25:
                        vad_feat = feat
                    all_feats.append(vad_feat)
                    for spec_aug_pipeline in self.spec_aug_pipelines:
                        all_feats.append(spec_aug_pipeline(vad_feat))

                labels = [label] * len(all_feats)
            except:
                traceback.print_exc()
                raise
            return all_feats, labels



        def __getitem__(self, index):
            if self._is_select_by_speaker:
                res = self._select_by_speaker(index)
            else:
                res = self._select_by_utt(index)

            return res

        def __len__(self):
            return len(self.utt_info) * self.repeat

        def __iter__(self):
            self._start = 0
            return self

        def __next__(self):
            if self._start < len(self):
                ret = self[self._start]
                self._start += 1
                return ret
            else:
                raise StopIteration

    return AudioDataset


if __name__ == "__main__":
    from sidt.utils.seed import seed_everything
    from torch.utils.data import DataLoader
    import torch
    import time
    seed_everything(0)

    train_scp_file = sys.argv[1]
    train_spk2utt = sys.argv[2]
    batch_size = 64
    augment_pipelines = [Tempo(), AddNoise(), Clicks(), Reverb()]

    def test_dataset(dataset):
        """
        Test function.

        Args:
            dataset: Paddle/pytorch Dataset
        """
        log.info("Dataset length = %d" % (len(train_dataset)))
        st = time.time()
        for idx, data in enumerate(dataset):
            end = time.time()
            log.info("Load %d batch. time = %.2fms" % (idx, (end - st) * 1000))
            if idx >= 5:
                break
            st = end

    try:
        import torch
        log.info("Test torch dataset")
        train_dataset = create_audio_dataset(torch.utils.data.Dataset)(train_scp_file, train_spk2utt,
                                                                       augment_pipelines=augment_pipelines, repeat=100)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=0,
                                                   collate_fn=train_dataset.collate_fn)
        test_dataset(train_loader)

        dev_dataset = create_audio_dataset(torch.utils.data.Dataset)(train_scp_file, train_spk2utt,
                                                                         select_by_speaker=False)
        dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=1,
                                                   collate_fn=train_dataset.collate_fn)
        test_dataset(dev_loader)
    except:
        log.warning("Pytorch is not available")
        raise

    try:
        import paddle
        paddle.disable_static()
        place = paddle.CPUPlace()
        log.info("Test paddlepaddle dataset")
        train_dataset = create_audio_dataset(paddle.io.Dataset)(train_scp_file, train_spk2utt,
                                                                augment_pipelines=augment_pipelines, repeat=100)
        train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                            return_list=True, shuffle=False,
                                            places=place, num_workers=1,
                                            collate_fn=train_dataset.collate_fn)
        test_dataset(train_loader)

        dev_dataset = create_audio_dataset(paddle.io.Dataset)(train_scp_file, train_spk2utt,
                                                                  select_by_speaker=False)
        dev_loader = paddle.io.DataLoader(dataset=dev_dataset, batch_size=batch_size,
                                            return_list=True, shuffle=False,
                                            places=place, num_workers=1,
                                            collate_fn=train_dataset.collate_fn)
        test_dataset(dev_loader)
    except:
        log.warning("Paddle is not available")
        raise
