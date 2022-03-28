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
# this is modified from SpeechBrain
# https://github.com/speechbrain/speechbrain/blob/085be635c07f16d42cd1295045bc46c407f1e15b/speechbrain/lobes/augment.py
import math
from typing import List

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleaudio.datasets.rirs_noises import OpenRIRNoise
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.io.signal_processing import compute_amplitude
from paddlespeech.vector.io.signal_processing import convolve1d
from paddlespeech.vector.io.signal_processing import dB_to_amplitude
from paddlespeech.vector.io.signal_processing import notch_filter
from paddlespeech.vector.io.signal_processing import reverberate

logger = Log(__name__).getlog()


# TODO: Complete type-hint and doc string.
class DropFreq(nn.Layer):
    def __init__(
            self,
            drop_freq_low=1e-14,
            drop_freq_high=1,
            drop_count_low=1,
            drop_count_high=2,
            drop_width=0.05,
            drop_prob=1, ):
        super(DropFreq, self).__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.drop_prob = drop_prob

    def forward(self, waveforms):
        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.clone()
        if paddle.rand([1]) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(-1)

        # Pick number of frequencies to drop
        drop_count = paddle.randint(
            low=self.drop_count_low, high=self.drop_count_high + 1, shape=[1])

        # Pick a frequency to drop
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequency = (
            paddle.rand([drop_count]) * drop_range + self.drop_freq_low)

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = paddle.zeros([1, filter_length, 1])
        drop_filter[0, pad, 0] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(frequency, filter_length,
                                        self.drop_width)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze(-1)


class DropChunk(nn.Layer):
    def __init__(
            self,
            drop_length_low=100,
            drop_length_high=1000,
            drop_count_low=1,
            drop_count_high=10,
            drop_start=0,
            drop_end=None,
            drop_prob=1,
            noise_factor=0.0, ):
        super(DropChunk, self).__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.drop_prob = drop_prob
        self.noise_factor = noise_factor

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        # Make sure the length doesn't exceed end - start
        if drop_end is not None and drop_end >= 0:
            if drop_start > drop_end:
                raise ValueError("Low limit must not be more than high limit")

            drop_range = drop_end - drop_start
            self.drop_length_low = min(drop_length_low, drop_range)
            self.drop_length_high = min(drop_length_high, drop_range)

    def forward(self, waveforms, lengths):
        # Reading input list
        lengths = (lengths * waveforms.shape[1]).astype('int64')
        batch_size = waveforms.shape[0]
        dropped_waveform = waveforms.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if paddle.rand([1]) > self.drop_prob:
            return dropped_waveform

        # Store original amplitude for computing white noise amplitude
        clean_amplitude = compute_amplitude(waveforms, lengths.unsqueeze(1))

        # Pick a number of times to drop
        drop_times = paddle.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            shape=[batch_size], )

        # Iterate batch to set mask
        for i in range(batch_size):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            length = paddle.randint(
                low=self.drop_length_low,
                high=self.drop_length_high + 1,
                shape=[drop_times[i]], )

            # Compute range of starting locations
            start_min = self.drop_start
            if start_min < 0:
                start_min += lengths[i]
            start_max = self.drop_end
            if start_max is None:
                start_max = lengths[i]
            if start_max < 0:
                start_max += lengths[i]
            start_max = max(0, start_max - length.max())

            # Pick starting locations
            start = paddle.randint(
                low=start_min,
                high=start_max + 1,
                shape=[drop_times[i]], )

            end = start + length

            # Update waveform
            if not self.noise_factor:
                for j in range(drop_times[i]):
                    if start[j] < end[j]:
                        dropped_waveform[i, start[j]:end[j]] = 0.0
            else:
                # Uniform distribution of -2 to +2 * avg amplitude should
                # preserve the average for normalization
                noise_max = 2 * clean_amplitude[i] * self.noise_factor
                for j in range(drop_times[i]):
                    # zero-center the noise distribution
                    noise_vec = paddle.rand([length[j]], dtype='float32')

                    noise_vec = 2 * noise_max * noise_vec - noise_max
                    dropped_waveform[i, int(start[j]):int(end[j])] = noise_vec

        return dropped_waveform


class Resample(nn.Layer):
    def __init__(
            self,
            orig_freq=16000,
            new_freq=16000,
            lowpass_filter_width=6, ):
        super(Resample, self).__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width

        # Compute rate for striding
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

    def _compute_strides(self):
        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, waveforms):
        if not hasattr(self, "first_indices"):
            self._indices_and_weights(waveforms)

        # Don't do anything if the frequencies are the same
        if self.orig_freq == self.new_freq:
            return waveforms

        unsqueezed = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3:
            waveforms = waveforms.transpose([0, 2, 1])
        else:
            raise ValueError("Input must be 2 or 3 dimensions")

        # Do resampling
        resampled_waveform = self._perform_resample(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(1)
        else:
            resampled_waveform = resampled_waveform.transpose([0, 2, 1])

        return resampled_waveform

    def _perform_resample(self, waveforms):
        # Compute output size and initialize
        batch_size, num_channels, wave_len = waveforms.shape
        window_size = self.weights.shape[1]
        tot_output_samp = self._output_samples(wave_len)
        resampled_waveform = paddle.zeros((batch_size, num_channels,
                                           tot_output_samp))

        # eye size: (num_channels, num_channels, 1)
        eye = paddle.eye(num_channels).unsqueeze(2)

        # Iterate over the phases in the polyphase filter
        for i in range(self.first_indices.shape[0]):
            wave_to_conv = waveforms
            first_index = int(self.first_indices[i].item())
            if first_index >= 0:
                # trim the signal as the filter will not be applied
                # before the first_index
                wave_to_conv = wave_to_conv[:, :, first_index:]

            # pad the right of the signal to allow partial convolutions
            # meaning compute values for partial windows (e.g. end of the
            # window is outside the signal length)
            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            wave_to_conv = paddle.nn.functional.pad(
                wave_to_conv, [left_padding, right_padding], data_format='NCL')
            conv_wave = paddle.nn.functional.conv1d(
                x=wave_to_conv,
                # weight=self.weights[i].repeat(num_channels, 1, 1),
                weight=self.weights[i].expand((num_channels, 1, -1)),
                stride=self.conv_stride,
                groups=num_channels, )

            # we want conv_wave[:, i] to be at
            # output[:, i + n*conv_transpose_stride]
            dilated_conv_wave = paddle.nn.functional.conv1d_transpose(
                conv_wave, eye, stride=self.conv_transpose_stride)

            # pad dilated_conv_wave so it reaches the output length if needed.
            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.shape[-1]
            right_padding = max(0, tot_output_samp - previous_padding)
            dilated_conv_wave = paddle.nn.functional.pad(
                dilated_conv_wave, [left_padding, right_padding],
                data_format='NCL')
            dilated_conv_wave = dilated_conv_wave[:, :, :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        samp_in = int(self.orig_freq)
        samp_out = int(self.new_freq)

        tick_freq = abs(samp_in * samp_out) // math.gcd(samp_in, samp_out)
        ticks_per_input_period = tick_freq // samp_in

        # work out the number of ticks in the time interval
        # [ 0, input_num_samp/samp_in ).
        interval_length = input_num_samp * ticks_per_input_period
        if interval_length <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_out

        # Get the last output-sample in the closed interval,
        # i.e. replacing [ ) with [ ]. Note: integer division rounds down.
        # See http://en.wikipedia.org/wiki/Interval_(mathematics) for an
        # explanation of the notation.
        last_output_samp = interval_length // ticks_per_output_period

        # We need the last output-sample in the open interval, so if it
        # takes us to the end of the interval exactly, subtract one.
        if last_output_samp * ticks_per_output_period == interval_length:
            last_output_samp -= 1

        # First output-sample index is zero, so the number of output samples
        # is the last output-sample plus one.
        num_output_samp = last_output_samp + 1

        return num_output_samp

    def _indices_and_weights(self, waveforms):
        # Lowpass filter frequency depends on smaller of two frequencies
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        output_t = paddle.arange(start=0.0, end=self.output_samples)
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = paddle.ceil(min_t * self.orig_freq)
        max_input_index = paddle.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        j = paddle.arange(max_weight_width, dtype='float32')
        input_index = min_input_index.unsqueeze(1) + j.unsqueeze(0)
        delta_t = (input_index / self.orig_freq) - output_t.unsqueeze(1)

        weights = paddle.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().less_than(
            paddle.to_tensor(window_width))

        # raised-cosine (Hanning) window with width `window_width`
        weights[inside_window_indices] = 0.5 * (1 + paddle.cos(
            2 * math.pi * lowpass_cutoff / self.lowpass_filter_width *
            delta_t.masked_select(inside_window_indices)))

        t_eq_zero_indices = delta_t.equal(paddle.zeros_like(delta_t))
        t_not_eq_zero_indices = delta_t.not_equal(paddle.zeros_like(delta_t))

        # sinc filter function
        weights = paddle.where(
            t_not_eq_zero_indices,
            weights * paddle.sin(2 * math.pi * lowpass_cutoff * delta_t) /
            (math.pi * delta_t), weights)

        # limit of the function at t = 0
        weights = paddle.where(t_eq_zero_indices, weights * 2 * lowpass_cutoff,
                               weights)

        # size (output_samples, max_weight_width)
        weights /= self.orig_freq

        self.first_indices = min_input_index
        self.weights = weights


class SpeedPerturb(nn.Layer):
    def __init__(
            self,
            orig_freq,
            speeds=[90, 100, 110],
            perturb_prob=1.0, ):
        super(SpeedPerturb, self).__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.perturb_prob = perturb_prob

        # Initialize index of perturbation
        self.samp_index = 0

        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:
            config = {
                "orig_freq": self.orig_freq,
                "new_freq": self.orig_freq * speed // 100,
            }
            self.resamplers.append(Resample(**config))

    def forward(self, waveform):
        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if paddle.rand([1]) > self.perturb_prob:
            return waveform.clone()

        # Perform a random perturbation
        self.samp_index = paddle.randint(len(self.speeds), shape=[1]).item()
        perturbed_waveform = self.resamplers[self.samp_index](waveform)

        return perturbed_waveform


class AddNoise(nn.Layer):
    def __init__(
            self,
            noise_dataset=None,  # None for white noise
            num_workers=0,
            snr_low=0,
            snr_high=0,
            mix_prob=1.0,
            start_index=None,
            normalize=False, ):
        super(AddNoise, self).__init__()

        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob
        self.start_index = start_index
        self.normalize = normalize
        self.noise_dataset = noise_dataset
        self.noise_dataloader = None

    def forward(self, waveforms, lengths=None):
        if lengths is None:
            lengths = paddle.ones([len(waveforms)])

        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).astype('int64').unsqueeze(1)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if paddle.rand([1]) > self.mix_prob:
            return noisy_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveforms, lengths)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = paddle.rand((len(waveforms), 1))
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        noisy_waveform *= 1 - noise_amplitude_factor

        # Loop through clean samples and create mixture
        if self.noise_dataset is None:
            white_noise = paddle.normal(shape=waveforms.shape)
            noisy_waveform += new_noise_amplitude * white_noise
        else:
            tensor_length = waveforms.shape[1]
            noise_waveform, noise_length = self._load_noise(
                lengths,
                tensor_length, )

            # Rescale and add
            noise_amplitude = compute_amplitude(noise_waveform, noise_length)
            noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)
            noisy_waveform += noise_waveform

        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = paddle.max(
                paddle.abs(noisy_waveform), axis=1, keepdim=True)
            noisy_waveform = noisy_waveform / abs_max.clip(min=1.0)

        return noisy_waveform

    def _load_noise(self, lengths, max_length):
        """
        Load a batch of noises

        args
        lengths(Paddle.Tensor): Num samples of waveforms with shape (N, 1).
        max_length(int): Width of a batch.
        """
        lengths = lengths.squeeze(1)
        batch_size = len(lengths)

        # Load a noise batch
        if self.noise_dataloader is None:

            def noise_collate_fn(batch):
                def pad(x, target_length, mode='constant', **kwargs):
                    x = np.asarray(x)
                    w = target_length - x.shape[0]
                    assert w >= 0, f'Target length {target_length} is less than origin length {x.shape[0]}'
                    return np.pad(x, [0, w], mode=mode, **kwargs)

                ids = [item['id'] for item in batch]
                lengths = np.asarray([item['feat'].shape[0] for item in batch])
                waveforms = list(
                    map(lambda x: pad(x, max(max_length, lengths.max().item())),
                        [item['feat'] for item in batch]))
                waveforms = np.stack(waveforms)
                return {'ids': ids, 'feats': waveforms, 'lengths': lengths}

            # Create noise data loader.
            self.noise_dataloader = paddle.io.DataLoader(
                self.noise_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=noise_collate_fn,
                return_list=True, )
            self.noise_data = iter(self.noise_dataloader)

        noise_batch, noise_len = self._load_noise_batch_of_size(batch_size)

        # Select a random starting location in the waveform
        start_index = self.start_index
        if self.start_index is None:
            start_index = 0
            max_chop = (noise_len - lengths).min().clip(min=1)
            start_index = paddle.randint(high=max_chop, shape=[1])

        # Truncate noise_batch to max_length
        noise_batch = noise_batch[:, start_index:start_index + max_length]
        noise_len = (noise_len - start_index).clip(max=max_length).unsqueeze(1)
        return noise_batch, noise_len

    def _load_noise_batch_of_size(self, batch_size):
        """Concatenate noise batches, then chop to correct size"""
        noise_batch, noise_lens = self._load_noise_batch()

        # Expand
        while len(noise_batch) < batch_size:
            noise_batch = paddle.concat((noise_batch, noise_batch))
            noise_lens = paddle.concat((noise_lens, noise_lens))

        # Contract
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            noise_lens = noise_lens[:batch_size]

        return noise_batch, noise_lens

    def _load_noise_batch(self):
        """Load a batch of noises, restarting iteration if necessary."""
        try:
            batch = next(self.noise_data)
        except StopIteration:
            self.noise_data = iter(self.noise_dataloader)
            batch = next(self.noise_data)

        noises, lens = batch['feats'], batch['lengths']
        return noises, lens


class AddReverb(nn.Layer):
    def __init__(
            self,
            rir_dataset,
            reverb_prob=1.0,
            rir_scale_factor=1.0,
            num_workers=0, ):
        super(AddReverb, self).__init__()
        self.rir_dataset = rir_dataset
        self.reverb_prob = reverb_prob
        self.rir_scale_factor = rir_scale_factor

        # Create rir data loader.
        def rir_collate_fn(batch):
            def pad(x, target_length, mode='constant', **kwargs):
                x = np.asarray(x)
                w = target_length - x.shape[0]
                assert w >= 0, f'Target length {target_length} is less than origin length {x.shape[0]}'
                return np.pad(x, [0, w], mode=mode, **kwargs)

            ids = [item['id'] for item in batch]
            lengths = np.asarray([item['feat'].shape[0] for item in batch])
            waveforms = list(
                map(lambda x: pad(x, lengths.max().item()),
                    [item['feat'] for item in batch]))
            waveforms = np.stack(waveforms)
            return {'ids': ids, 'feats': waveforms, 'lengths': lengths}

        self.rir_dataloader = paddle.io.DataLoader(
            self.rir_dataset,
            collate_fn=rir_collate_fn,
            num_workers=num_workers,
            shuffle=True,
            return_list=True, )

        self.rir_data = iter(self.rir_dataloader)

    def forward(self, waveforms, lengths=None):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if lengths is None:
            lengths = paddle.ones([len(waveforms)])

        # Don't add reverb (return early) 1-`reverb_prob` portion of the time
        if paddle.rand([1]) > self.reverb_prob:
            return waveforms.clone()

        # Add channels dimension if necessary
        channel_added = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(-1)
            channel_added = True

        # Load and prepare RIR
        rir_waveform = self._load_rir()

        # Compress or dilate RIR
        if self.rir_scale_factor != 1:
            rir_waveform = F.interpolate(
                rir_waveform.transpose([0, 2, 1]),
                scale_factor=self.rir_scale_factor,
                mode="linear",
                align_corners=False,
                data_format='NCW', )
            # (N, C, L) -> (N, L, C)
            rir_waveform = rir_waveform.transpose([0, 2, 1])

        rev_waveform = reverberate(
            waveforms,
            rir_waveform,
            self.rir_dataset.sample_rate,
            rescale_amp="avg")

        # Remove channels dimension if added
        if channel_added:
            return rev_waveform.squeeze(-1)

        return rev_waveform

    def _load_rir(self):
        try:
            batch = next(self.rir_data)
        except StopIteration:
            self.rir_data = iter(self.rir_dataloader)
            batch = next(self.rir_data)

        rir_waveform = batch['feats']

        # Make sure RIR has correct channels
        if len(rir_waveform.shape) == 2:
            rir_waveform = rir_waveform.unsqueeze(-1)

        return rir_waveform


class AddBabble(nn.Layer):
    def __init__(
            self,
            speaker_count=3,
            snr_low=0,
            snr_high=0,
            mix_prob=1, ):
        super(AddBabble, self).__init__()
        self.speaker_count = speaker_count
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob

    def forward(self, waveforms, lengths=None):
        if lengths is None:
            lengths = paddle.ones([len(waveforms)])

        babbled_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)
        batch_size = len(waveforms)

        # Don't mix (return early) 1-`mix_prob` portion of the batches
        if paddle.rand([1]) > self.mix_prob:
            return babbled_waveform

        # Pick an SNR and use it to compute the mixture amplitude factors
        clean_amplitude = compute_amplitude(waveforms, lengths)
        SNR = paddle.rand((batch_size, 1))
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        babbled_waveform *= 1 - noise_amplitude_factor

        # For each speaker in the mixture, roll and add
        babble_waveform = waveforms.roll((1, ), axis=0)
        babble_len = lengths.roll((1, ), axis=0)
        for i in range(1, self.speaker_count):
            babble_waveform += waveforms.roll((1 + i, ), axis=0)
            babble_len = paddle.concat(
                [babble_len, babble_len.roll((1, ), axis=0)], axis=-1).max(
                    axis=-1, keepdim=True)

        # Rescale and add to mixture
        babble_amplitude = compute_amplitude(babble_waveform, babble_len)
        babble_waveform *= new_noise_amplitude / (babble_amplitude + 1e-14)
        babbled_waveform += babble_waveform

        return babbled_waveform


class TimeDomainSpecAugment(nn.Layer):
    def __init__(
            self,
            perturb_prob=1.0,
            drop_freq_prob=1.0,
            drop_chunk_prob=1.0,
            speeds=[95, 100, 105],
            sample_rate=16000,
            drop_freq_count_low=0,
            drop_freq_count_high=3,
            drop_chunk_count_low=0,
            drop_chunk_count_high=5,
            drop_chunk_length_low=1000,
            drop_chunk_length_high=2000,
            drop_chunk_noise_factor=0, ):
        super(TimeDomainSpecAugment, self).__init__()
        self.speed_perturb = SpeedPerturb(
            perturb_prob=perturb_prob,
            orig_freq=sample_rate,
            speeds=speeds, )
        self.drop_freq = DropFreq(
            drop_prob=drop_freq_prob,
            drop_count_low=drop_freq_count_low,
            drop_count_high=drop_freq_count_high, )
        self.drop_chunk = DropChunk(
            drop_prob=drop_chunk_prob,
            drop_count_low=drop_chunk_count_low,
            drop_count_high=drop_chunk_count_high,
            drop_length_low=drop_chunk_length_low,
            drop_length_high=drop_chunk_length_high,
            noise_factor=drop_chunk_noise_factor, )

    def forward(self, waveforms, lengths=None):
        if lengths is None:
            lengths = paddle.ones([len(waveforms)])

        with paddle.no_grad():
            # Augmentation
            waveforms = self.speed_perturb(waveforms)
            waveforms = self.drop_freq(waveforms)
            waveforms = self.drop_chunk(waveforms, lengths)

        return waveforms


class EnvCorrupt(nn.Layer):
    def __init__(
            self,
            reverb_prob=1.0,
            babble_prob=1.0,
            noise_prob=1.0,
            rir_dataset=None,
            noise_dataset=None,
            num_workers=0,
            babble_speaker_count=0,
            babble_snr_low=0,
            babble_snr_high=0,
            noise_snr_low=0,
            noise_snr_high=0,
            rir_scale_factor=1.0, ):
        super(EnvCorrupt, self).__init__()

        # Initialize corrupters
        if rir_dataset is not None and reverb_prob > 0.0:
            self.add_reverb = AddReverb(
                rir_dataset=rir_dataset,
                num_workers=num_workers,
                reverb_prob=reverb_prob,
                rir_scale_factor=rir_scale_factor, )

        if babble_speaker_count > 0 and babble_prob > 0.0:
            self.add_babble = AddBabble(
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
                mix_prob=babble_prob, )

        if noise_dataset is not None and noise_prob > 0.0:
            self.add_noise = AddNoise(
                noise_dataset=noise_dataset,
                num_workers=num_workers,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
                mix_prob=noise_prob, )

    def forward(self, waveforms, lengths=None):
        if lengths is None:
            lengths = paddle.ones([len(waveforms)])

        # Augmentation
        with paddle.no_grad():
            if hasattr(self, "add_reverb"):
                try:
                    waveforms = self.add_reverb(waveforms, lengths)
                except Exception:
                    pass
            if hasattr(self, "add_babble"):
                waveforms = self.add_babble(waveforms, lengths)
            if hasattr(self, "add_noise"):
                waveforms = self.add_noise(waveforms, lengths)

        return waveforms


def build_augment_pipeline(target_dir=None) -> List[paddle.nn.Layer]:
    """build augment pipeline
    Note: this pipeline cannot be used in the paddle.DataLoader

    Returns:
        List[paddle.nn.Layer]: all augment process
    """
    logger.info("start to build the augment pipeline")
    noise_dataset = OpenRIRNoise('noise', target_dir=target_dir)
    rir_dataset = OpenRIRNoise('rir', target_dir=target_dir)

    wavedrop = TimeDomainSpecAugment(
        sample_rate=16000,
        speeds=[100], )
    speed_perturb = TimeDomainSpecAugment(
        sample_rate=16000,
        speeds=[95, 100, 105], )
    add_noise = EnvCorrupt(
        noise_dataset=noise_dataset,
        reverb_prob=0.0,
        noise_prob=1.0,
        noise_snr_low=0,
        noise_snr_high=15,
        rir_scale_factor=1.0, )
    add_rev = EnvCorrupt(
        rir_dataset=rir_dataset,
        reverb_prob=1.0,
        noise_prob=0.0,
        rir_scale_factor=1.0, )
    add_rev_noise = EnvCorrupt(
        noise_dataset=noise_dataset,
        rir_dataset=rir_dataset,
        reverb_prob=1.0,
        noise_prob=1.0,
        noise_snr_low=0,
        noise_snr_high=15,
        rir_scale_factor=1.0, )

    return [wavedrop, speed_perturb, add_noise, add_rev, add_rev_noise]


def waveform_augment(waveforms: paddle.Tensor,
                     augment_pipeline: List[paddle.nn.Layer]) -> paddle.Tensor:
    """process the augment pipeline and return all the waveforms

    Args:
        waveforms (paddle.Tensor): original batch waveform
        augment_pipeline (List[paddle.nn.Layer]): agument pipeline process

    Returns:
        paddle.Tensor: all the audio waveform including the original waveform and augmented waveform
    """
    # stage 0: store the original waveforms
    waveforms_aug_list = [waveforms]

    # augment the original batch waveform
    for aug in augment_pipeline:
        # stage 1: augment the data
        waveforms_aug = aug(waveforms)  # (N, L)
        if waveforms_aug.shape[1] >= waveforms.shape[1]:
            # Trunc
            waveforms_aug = waveforms_aug[:, :waveforms.shape[1]]
        else:
            # Pad
            lengths_to_pad = waveforms.shape[1] - waveforms_aug.shape[1]
            waveforms_aug = F.pad(
                waveforms_aug.unsqueeze(-1), [0, lengths_to_pad],
                data_format='NLC').squeeze(-1)
        # stage 2: append the augmented waveform into the list
        waveforms_aug_list.append(waveforms_aug)

    # get the all the waveforms
    return paddle.concat(waveforms_aug_list, axis=0)
