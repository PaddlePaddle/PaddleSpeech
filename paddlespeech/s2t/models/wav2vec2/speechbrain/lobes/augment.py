import os
import paddle
import speechbrain as sb
from speechbrain.processing.speech_augmentation import (
    SpeedPerturb,
    DropFreq,
    DropChunk,
)


class TimeDomainSpecAugment(paddle.nn.Layer):
    """A time-domain approximation of the SpecAugment algorithm.
    This augmentation module implements three augmentations in
    the time-domain.
     1. Drop chunks of the audio (zero amplitude or white noise)
     2. Drop frequency bands (with band-drop filters)
     3. Speed peturbation (via resampling to slightly different rate)
    Arguments
    ---------
    perturb_prob : float from 0 to 1
        The probability that a batch will have speed perturbation applied.
    drop_freq_prob : float from 0 to 1
        The probability that a batch will have frequencies dropped.
    drop_chunk_prob : float from 0 to 1
        The probability that a batch will have chunks dropped.
    speeds : list of ints
        A set of different speeds to use to perturb each batch.
        See ``speechbrain.processing.speech_augmentation.SpeedPerturb``
    sample_rate : int
        Sampling rate of the input waveforms.
    drop_freq_count_low : int
        Lowest number of frequencies that could be dropped.
    drop_freq_count_high : int
        Highest number of frequencies that could be dropped.
    drop_chunk_count_low : int
        Lowest number of chunks that could be dropped.
    drop_chunk_count_high : int
        Highest number of chunks that could be dropped.
    drop_chunk_length_low : int
        Lowest length of chunks that could be dropped.
    drop_chunk_length_high : int
        Highest length of chunks that could be dropped.
    drop_chunk_noise_factor : float
        The noise factor used to scale the white noise inserted, relative to
        the average amplitude of the utterance. Default 0 (no noise inserted).
    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = TimeDomainSpecAugment(speeds=[80])
    >>> feats = feature_maker(inputs, torch.ones(10))
    >>> feats.shape
    torch.Size([10, 12800])
    """

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
        drop_chunk_noise_factor=0,
    ):
        super().__init__()
        self.speed_perturb = SpeedPerturb(
            perturb_prob=perturb_prob, orig_freq=sample_rate, speeds=speeds
        )
        self.drop_freq = DropFreq(
            drop_prob=drop_freq_prob,
            drop_count_low=drop_freq_count_low,
            drop_count_high=drop_freq_count_high,
        )
        self.drop_chunk = DropChunk(
            drop_prob=drop_chunk_prob,
            drop_count_low=drop_chunk_count_low,
            drop_count_high=drop_chunk_count_high,
            drop_length_low=drop_chunk_length_low,
            drop_length_high=drop_chunk_length_high,
            noise_factor=drop_chunk_noise_factor,
        )

    def forward(self, waveforms, lengths):
        """Returns the distorted waveforms.
        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort
        """
        # Augmentation
        with paddle.no_grad():
            waveforms = self.speed_perturb(waveforms)
            waveforms = self.drop_freq(waveforms)
            waveforms = self.drop_chunk(waveforms, lengths)

        return 


class DropFreq(torch.nn.Module):
    """This class drops a random frequency from the signal.
    The purpose of this class is to teach models to learn to rely on all parts
    of the signal, not just a few frequency bands.
    Arguments
    ---------
    drop_freq_low : float
        The low end of frequencies that can be dropped,
        as a fraction of the sampling rate / 2.
    drop_freq_high : float
        The high end of frequencies that can be
        dropped, as a fraction of the sampling rate / 2.
    drop_count_low : int
        The low end of number of frequencies that could be dropped.
    drop_count_high : int
        The high end of number of frequencies that could be dropped.
    drop_width : float
        The width of the frequency band to drop, as
        a fraction of the sampling_rate / 2.
    drop_prob : float
        The probability that the batch of signals will  have a frequency
        dropped. By default, every batch has frequencies dropped.
    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> dropper = DropFreq()
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> dropped_signal = dropper(signal.unsqueeze(0))
    """

    def __init__(
        self,
        drop_freq_low=1e-14,
        drop_freq_high=1,
        drop_count_low=1,
        drop_count_high=2,
        drop_width=0.05,
        drop_prob=1,
    ):
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.drop_prob = drop_prob

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.clone()
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(-1)

        # Pick number of frequencies to drop
        drop_count = torch.randint(
            low=self.drop_count_low, high=self.drop_count_high + 1, size=(1,),
        )

        # Pick a frequency to drop
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequency = (
            torch.rand(drop_count) * drop_range + self.drop_freq_low
        )

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = torch.zeros(1, filter_length, 1, device=waveforms.device)
        drop_filter[0, pad, 0] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(
                frequency, filter_length, self.drop_width,
            ).to(waveforms.device)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze(-1)

class DropChunk(torch.nn.Module):
    """This class drops portions of the input signal.
    Using `DropChunk` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.
    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to set the
        signal to zero, in samples.
    drop_length_high : int
        The high end of lengths for which to set the
        signal to zero, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped to zero.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped to zero.
    drop_start : int
        The first index for which dropping will be allowed.
    drop_end : int
        The last index for which dropping will be allowed.
    drop_prob : float
        The probability that the batch of signals will
        have a portion dropped. By default, every batch
        has portions dropped.
    noise_factor : float
        The factor relative to average amplitude of an utterance
        to use for scaling the white noise inserted. 1 keeps
        the average amplitude the same, while 0 inserts all 0's.
    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> dropper = DropChunk(drop_start=100, drop_end=200, noise_factor=0.)
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    >>> length = torch.ones(1)
    >>> dropped_signal = dropper(signal, length)
    >>> float(dropped_signal[:, 150])
    0.0
    """

    def __init__(
        self,
        drop_length_low=100,
        drop_length_high=1000,
        drop_count_low=1,
        drop_count_high=10,
        drop_start=0,
        drop_end=None,
        drop_prob=1,
        noise_factor=0.0,
    ):
        super().__init__()
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
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.
        Returns
        -------
        Tensor of shape `[batch, time]` or
            `[batch, time, channels]`
        """

        # Reading input list
        lengths = (lengths * waveforms.size(1)).long()
        batch_size = waveforms.size(0)
        dropped_waveform = waveforms.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Store original amplitude for computing white noise amplitude
        clean_amplitude = compute_amplitude(waveforms, lengths.unsqueeze(1))

        # Pick a number of times to drop
        drop_times = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(batch_size,),
        )

        # Iterate batch to set mask
        for i in range(batch_size):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            length = torch.randint(
                low=self.drop_length_low,
                high=self.drop_length_high + 1,
                size=(drop_times[i],),
            )

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
            start = torch.randint(
                low=start_min, high=start_max + 1, size=(drop_times[i],),
            )

            end = start + length

            # Update waveform
            if not self.noise_factor:
                for j in range(drop_times[i]):
                    dropped_waveform[i, start[j] : end[j]] = 0.0
            else:
                # Uniform distribution of -2 to +2 * avg amplitude should
                # preserve the average for normalization
                noise_max = 2 * clean_amplitude[i] * self.noise_factor
                for j in range(drop_times[i]):
                    # zero-center the noise distribution
                    noise_vec = torch.rand(length[j], device=waveforms.device)
                    noise_vec = 2 * noise_max * noise_vec - noise_max
                    dropped_waveform[i, start[j] : end[j]] = noise_vec

        return 