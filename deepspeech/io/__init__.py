# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from paddle.io import DataLoader

from deepspeech.io.collator import SpeechCollator
from deepspeech.io.dataset import ManifestDataset
from deepspeech.io.sampler import SortagradBatchSampler
from deepspeech.io.sampler import SortagradDistributedBatchSampler


def create_dataloader(manifest_path,
                      unit_type,
                      vocab_filepath,
                      mean_std_filepath,
                      spm_model_prefix,
                      augmentation_config='{}',
                      max_input_len=float('inf'),
                      min_input_len=0.0,
                      max_output_len=float('inf'),
                      min_output_len=0.0,
                      max_output_input_ratio=float('inf'),
                      min_output_input_ratio=0.0,
                      stride_ms=10.0,
                      window_ms=20.0,
                      max_freq=None,
                      specgram_type='linear',
                      feat_dim=None,
                      delta_delta=False,
                      use_dB_normalization=True,
                      random_seed=0,
                      keep_transcription_text=False,
                      is_training=False,
                      batch_size=1,
                      num_workers=0,
                      sortagrad=False,
                      shuffle_method=None,
                      dist=False):

    dataset = ManifestDataset(
        manifest_path=manifest_path,
        unit_type=unit_type,
        vocab_filepath=vocab_filepath,
        mean_std_filepath=mean_std_filepath,
        spm_model_prefix=spm_model_prefix,
        augmentation_config=augmentation_config,
        max_input_len=max_input_len,
        min_input_len=min_input_len,
        max_output_len=max_output_len,
        min_output_len=min_output_len,
        max_output_input_ratio=max_output_input_ratio,
        min_output_input_ratio=min_output_input_ratio,
        stride_ms=stride_ms,
        window_ms=window_ms,
        max_freq=max_freq,
        specgram_type=specgram_type,
        feat_dim=feat_dim,
        delta_delta=delta_delta,
        use_dB_normalization=use_dB_normalization,
        random_seed=random_seed,
        keep_transcription_text=keep_transcription_text)

    if dist:
        batch_sampler = SortagradDistributedBatchSampler(
            dataset,
            batch_size,
            num_replicas=None,
            rank=None,
            shuffle=is_training,
            drop_last=is_training,
            sortagrad=is_training,
            shuffle_method=shuffle_method)
    else:
        batch_sampler = SortagradBatchSampler(
            dataset,
            shuffle=is_training,
            batch_size=batch_size,
            drop_last=is_training,
            sortagrad=is_training,
            shuffle_method=shuffle_method)

    def padding_batch(batch,
                      padding_to=-1,
                      flatten=False,
                      keep_transcription_text=True):
        """	
        Padding audio features with zeros to make them have the same shape (or	
        a user-defined shape) within one bach.	

        If ``padding_to`` is -1, the maximun shape in the batch will be used	
        as the target shape for padding. Otherwise, `padding_to` will be the	
        target shape (only refers to the second axis).	

        If `flatten` is True, features will be flatten to 1darray.	
        """
        new_batch = []
        # get target shape	
        max_length = max([audio.shape[1] for audio, text in batch])
        if padding_to != -1:
            if padding_to < max_length:
                raise ValueError("If padding_to is not -1, it should be larger "
                                 "than any instance's shape in the batch")
            max_length = padding_to
        max_text_length = max([len(text) for audio, text in batch])
        # padding	
        padded_audios = []
        audio_lens = []
        texts, text_lens = [], []
        for audio, text in batch:
            padded_audio = np.zeros([audio.shape[0], max_length])
            padded_audio[:, :audio.shape[1]] = audio
            if flatten:
                padded_audio = padded_audio.flatten()
            padded_audios.append(padded_audio)
            audio_lens.append(audio.shape[1])

            padded_text = np.zeros([max_text_length])
            if keep_transcription_text:
                padded_text[:len(text)] = [ord(t) for t in text]  # string
            else:
                padded_text[:len(text)] = text  # ids
            texts.append(padded_text)
            text_lens.append(len(text))

        padded_audios = np.array(padded_audios).astype('float32')
        audio_lens = np.array(audio_lens).astype('int64')
        texts = np.array(texts).astype('int32')
        text_lens = np.array(text_lens).astype('int64')
        return padded_audios, audio_lens, texts, text_lens

    # collate_fn=functools.partial(padding_batch, keep_transcription_text=keep_transcription_text),
    collate_fn = SpeechCollator(keep_transcription_text=keep_transcription_text)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers)
    return loader
