"""Classes for mutating speech data for data augmentation.

This module provides classes that produce realistic distortions of speech
data for the purpose of training speech processing models. The list of
distortions includes adding noise, adding reverberation, changing speed,
and more. All the classes are of type `paddle.nn.Layer`. This gives the
possibility to have end-to-end differentiability and
backpropagate the gradient through them. In addition, all operations
are expected to be performed on the GPU (where available) for efficiency.

Authors
 * Peter Plantinga 2020
"""

# Importing libraries
import math
import paddle
import paddle.nn.functional as F
from speechbrain.dataio.legacy import ExtendedCSVDataset
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.processing.signal_processing import (
    compute_amplitude,
    dB_to_amplitude,
    convolve1d,
    notch_filter,
    reverberate,
)

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()
class AddNoise(paddle.nn.Layer):
    """This class additively combines a noise signal to the input signal.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    csv_keys : list, None, optional
        Default: None . One data entry for the noise data should be specified.
        If None, the csv file is expected to have only one data entry.
    sorting : str
        The order to iterate the csv file, from one of the
        following options: random, original, ascending, and descending.
    num_workers : int
        Number of workers in the DataLoader (See PyTorch DataLoader docs).
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise : bool
        If True, copy noise signals that are shorter than
        their corresponding clean signals so as to cover the whole clean
        signal. Otherwise, leave the noise un-padded.
    mix_prob : float
        The probability that a batch of signals will be mixed
        with a noise signal. By default, every batch is mixed with noise.
    start_index : int
        The index in the noise waveforms to start from. By default, chooses
        a random index in [0, len(noise) - len(waveforms)].
    normalize : bool
        If True, output noisy signals that exceed [-1,1] will be
        normalized to [-1,1].
    replacements : dict
        A set of string replacements to carry out in the
        csv file. Each time a key is found in the text, it will be replaced
        with the corresponding value.

    Example
    -------
    >>> import pytest
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> clean = signal.unsqueeze(0) # [batch, time, channels]
    >>> noisifier = AddNoise('samples/noise_samples/noise.csv')
    >>> noisy = noisifier(clean, paddle.ones(1))
    """

    def __init__(
        self,
        csv_file=None,
        csv_keys=None,
        sorting="random",
        num_workers=0,
        snr_low=0,
        snr_high=0,
        pad_noise=False,
        mix_prob=1.0,
        start_index=None,
        normalize=False,
        replacements={},
    ):
        super().__init__()

        self.csv_file = csv_file
        self.csv_keys = csv_keys
        self.sorting = sorting
        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob
        self.start_index = start_index
        self.normalize = normalize
        self.replacements = replacements

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
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if paddle.rand([1]).item() > self.mix_prob:
            return noisy_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveforms, lengths)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = paddle.rand([len(waveforms), 1])
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        noisy_waveform *= 1 - noise_amplitude_factor

        # Loop through clean samples and create mixture
        if self.csv_file is None:
            # 这里传入了tensor，没有其效果？？？？是一个bug吧？？
            # print("add the noise")
            white_noise = paddle.randn(shape=waveforms.shape)
            noisy_waveform += new_noise_amplitude * white_noise
        else:
            # 有csv内容，需要读取csv中的噪声音频
            tensor_length = waveforms.shape[1]
            # print("tensor length: {}".format(tensor_length))
            # print("length: {}".format(lengths))
            self._load_noise(
                lengths, tensor_length,
            )
            noise_waveform, noise_length = self._load_noise(
                lengths, tensor_length,
            )

            # Rescale and add
            noise_amplitude = compute_amplitude(noise_waveform, noise_length)
            noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)
            noisy_waveform += noise_waveform

        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = paddle.max(
                paddle.abs(noisy_waveform), dim=1, keepdim=True
            )
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform

    def _load_noise(self, lengths, max_length):
        """Load a batch of noises"""
        lengths = lengths.astype("int64").squeeze(1)
        batch_size = len(lengths)
        # Load a noise batch
        if not hasattr(self, "data_loader"):
            # Set parameters based on input
            # self.device = lengths.device
            # Create a data loader for the noise wavforms
            if self.csv_file is not None:
                dataset = ExtendedCSVDataset(
                    csvpath=self.csv_file,
                    output_keys=self.csv_keys,
                    sorting=self.sorting
                    if self.sorting != "random"
                    else "original",
                    replacements=self.replacements,
                )
                self.data_loader = make_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    shuffle=(self.sorting == "random"),
                )
                self.noise_data = iter(self.data_loader)

        # Load noise to correct device
        noise_batch, noise_len = self._load_noise_batch_of_size(batch_size)
        # noise_batch = noise_batch
        # noise_len = noise_len

        # Convert relative length to an index
        noise_len = (noise_len * noise_batch.shape[1]).astype("int64")

        # Ensure shortest wav can cover speech signal
        # WARNING: THIS COULD BE SLOW IF THERE ARE VERY SHORT NOISES
        if self.pad_noise:
            while paddle.any(noise_len < lengths):
                min_len = paddle.min(noise_len)
                prepend = noise_batch[:, :min_len]
                noise_batch = paddle.cat((prepend, noise_batch), axis=1)
                noise_len += min_len

        # Ensure noise batch is long enough
        elif noise_batch.shape[1] < max_length:
            # logger.info("max length: {}".format(max_length))
            padding = (0, max_length - noise_batch.shape[1])
            if len(noise_batch.shape) == 2:
                unsqueeze_flag = True
                noise_batch = noise_batch.unsqueeze(0)
            # logger.info("noise batch info: {}".format(noise_batch.shape))
            # logger.info("padding: {}".format(padding))
            noise_batch = paddle.nn.functional.pad(noise_batch, padding, data_format="NCL")
            if unsqueeze_flag:
                noise_batch = noise_batch.squeeze(0)

        # Select a random starting location in the waveform
        start_index = self.start_index
        if self.start_index is None:
            start_index = 0
            max_chop = (noise_len - lengths).min().clip(min=1)
            start_index = paddle.randint(
                high=max_chop, shape=[1,])

        # Truncate noise_batch to max_length
        noise_batch = noise_batch[:, start_index : start_index + max_length]
        noise_len = (noise_len - start_index).clip(max=max_length).unsqueeze(1)
        return noise_batch, noise_len

    def _load_noise_batch_of_size(self, batch_size):
        """Concatenate noise batches, then chop to correct size"""

        noise_batch, noise_lens = self._load_noise_batch()

        # Expand
        while len(noise_batch) < batch_size:
            added_noise, added_lens = self._load_noise_batch()
            noise_batch, noise_lens = AddNoise._concat_batch(
                noise_batch, noise_lens, added_noise, added_lens
            )

        # Contract
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            noise_lens = noise_lens[:batch_size]

        return noise_batch, noise_lens

    @staticmethod
    def _concat_batch(noise_batch, noise_lens, added_noise, added_lens):
        """Concatenate two noise batches of potentially different lengths"""

        # pad shorter batch to correct length
        noise_tensor_len = noise_batch.shape[1]
        added_tensor_len = added_noise.shape[1]
        pad = [0, abs(noise_tensor_len - added_tensor_len)]
        if noise_tensor_len > added_tensor_len:
            added_noise = added_noise.unsqueeze(0)
            # print("added_noise shape: {}".format(added_noise.shape))
            added_noise = paddle.nn.functional.pad(added_noise, pad,  data_format='NCL')
            added_noise = added_noise.squeeze(0)
            # print("added_noise shape: {}".format(added_noise.shape))
            added_lens = added_lens * added_tensor_len / noise_tensor_len
        else:
            noise_batch = noise_batch.unsqueeze(0)
            # print("noise batch shape: {}".format(noise_batch.shape))
            # print("pad shape: {}".format(pad))
            # 在paddle中至少3个维度，在torch中，两维时，默认最后一个是1
            noise_batch = paddle.nn.functional.pad(noise_batch, pad, data_format='NCL') 
            noise_batch = noise_batch.squeeze(0)
            # print("noise_batch shape: {}".format(noise_batch.shape))
            noise_lens = noise_lens * noise_tensor_len / added_tensor_len
        
        noise_batch = paddle.concat([noise_batch, added_noise])
        noise_lens = paddle.concat([noise_lens, added_lens])

        return noise_batch, noise_lens

    def _load_noise_batch(self):
        """Load a batch of noises, restarting iteration if necessary."""

        try:
            # Don't necessarily know the key
            # print("\nnoise data: {}".format(type(self.noise_data)))
            # print("next noise data: {}".format(type(next(self.noise_data))))
            noises, lens = next(self.noise_data)["batch_value"].at_position(0)
        except StopIteration:
            # print("load noise wave occurs error")
            self.noise_data = iter(self.data_loader)
            # print("noise data: {}".format(type(self.noise_data)))
            # next(self.noise_data)
            # print("next noise data: {}".format(type(next(self.noise_data))))
            # print("next noise data: {}".format(type(next(self.noise_data))))
            noises, lens = next(self.noise_data)["batch_value"].at_position(0)
        return noises, lens


class AddReverb(paddle.nn.Layer):
    """This class convolves an audio signal with an impulse response.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        impulse response files.
    sorting : str
        The order to iterate the csv file, from one of
        the following options: random, original, ascending, and descending.
    reverb_prob : float
        The chance that the audio signal will be reverbed.
        By default, every batch is reverbed.
    rir_scale_factor: float
        It compresses or dilates the given impulse response.
        If 0 < scale_factor < 1, the impulse response is compressed
        (less reverb), while if scale_factor > 1 it is dilated
        (more reverb).
    replacements : dict
        A set of string replacements to carry out in the
        csv file. Each time a key is found in the text, it will be replaced
        with the corresponding value.

    Example
    -------
    >>> import pytest
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> clean = signal.unsqueeze(0) # [batch, time, channels]
    >>> reverb = AddReverb('samples/rir_samples/rirs.csv')
    >>> reverbed = reverb(clean, paddle.ones(1))
    """

    def __init__(
        self,
        csv_file,
        sorting="random",
        reverb_prob=1.0,
        rir_scale_factor=1.0,
        replacements={},
    ):
        super().__init__()
        self.csv_file = csv_file
        self.sorting = sorting
        self.reverb_prob = reverb_prob
        self.replacements = replacements
        self.rir_scale_factor = rir_scale_factor

        # Create a data loader for the RIR waveforms
        print("sorting method: {}".format(self.sorting))
        dataset = ExtendedCSVDataset(
            csvpath=self.csv_file,
            sorting=self.sorting if self.sorting != "random" else "original",
            replacements=self.replacements,
        )
        self.data_loader = make_dataloader(
            dataset, shuffle=(self.sorting == "random")
        )
        self.rir_data = iter(self.data_loader)

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
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't add reverb (return early) 1-`reverb_prob` portion of the time
        if paddle.rand([1], dtype="float32") > self.reverb_prob:
            return waveforms.clone()

        # Add channels dimension if necessary
        channel_added = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(-1)
            channel_added = True

        # Convert length from ratio to number of indices
        # lengths = (lengths * waveforms.shape[1])[:, None, None]

        # Load and prepare RIR
        rir_waveform = self._load_rir(waveforms)

        # Compress or dilate RIR
        if self.rir_scale_factor != 1:
            rir_waveform = F.interpolate(
                rir_waveform.transpose(1, -1),
                scale_factor=self.rir_scale_factor,
                mode="linear",
                align_corners=False,
            )
            rir_waveform = rir_waveform.transpose(1, -1)

        rev_waveform = reverberate(waveforms, rir_waveform, rescale_amp="avg")

        # Remove channels dimension if added
        if channel_added:
            return rev_waveform.squeeze(-1)

        return rev_waveform

    def _load_rir(self, waveforms):
        try:
            rir_waveform, length = next(self.rir_data)["batch_value"].at_position(0)
        except StopIteration:
            self.rir_data = iter(self.data_loader)
            rir_waveform, length = next(self.rir_data)["batch_value"].at_position(0)

        # Make sure RIR has correct channels
        if len(rir_waveform.shape) == 2:
            rir_waveform = rir_waveform.unsqueeze(-1)

        # Make sure RIR has correct type and device
        rir_waveform = rir_waveform.astype(waveforms.dtype)
        return rir_waveform


class SpeedPerturb(paddle.nn.Layer):
    """Slightly speed up or slow down an audio signal.

    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"

    Arguments
    ---------
    orig_freq : int
        The frequency of the original signal.
    speeds : list
        The speeds that the signal should be changed to, as a percentage of the
        original signal (i.e. `speeds` is divided by 100 to get a ratio).
    perturb_prob : float
        The chance that the batch will be speed-
        perturbed. By default, every batch is perturbed.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> perturbator = SpeedPerturb(orig_freq=16000, speeds=[90])
    >>> clean = signal.unsqueeze(0)
    >>> perturbed = perturbator(clean)
    >>> clean.shape
    paddle.Size([1, 52173])
    >>> perturbed.shape
    paddle.Size([1, 46956])
    """

    def __init__(
        self, orig_freq, speeds=[90, 100, 110], perturb_prob=1.0,
    ):
        super().__init__()
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

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if paddle.rand([1]) > self.perturb_prob:
            return waveform.clone()

        # Perform a random perturbation
        self.samp_index = paddle.randint(len(self.speeds), shape=[1,])[0]
        perturbed_waveform = self.resamplers[self.samp_index](waveform)

        return perturbed_waveform


class Resample(paddle.nn.Layer):
    """This class resamples an audio signal using sinc-based interpolation.

    It is a modification of the `resample` function from torchaudio
    (https://pytorch.org/audio/transforms.html#resample)

    Arguments
    ---------
    orig_freq : int
        the sampling frequency of the input signal.
    new_freq : int
        the new sampling frequency after this operation is performed.
    lowpass_filter_width : int
        Controls the sharpness of the filter, larger numbers result in a
        sharper filter, but they are less efficient. Values from 4 to 10 are
        allowed.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    >>> resampler = Resample(orig_freq=16000, new_freq=8000)
    >>> resampled = resampler(signal)
    >>> signal.shape
    paddle.Size([1, 52173])
    >>> resampled.shape
    paddle.Size([1, 26087])
    """

    def __init__(
        self, orig_freq=16000, new_freq=16000, lowpass_filter_width=6,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width

        # Compute rate for striding
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

    def _compute_strides(self):
        """Compute the phases in polyphase filter.

        (almost directly from torchaudio.compliance.kaldi)
        """

        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, waveforms):
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
            waveforms = waveforms.transpose(1, 2)
        else:
            raise ValueError("Input must be 2 or 3 dimensions")

        # Do resampling
        resampled_waveform = self._perform_resample(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(1)
        else:
            resampled_waveform = resampled_waveform.transpose(1, 2)

        return resampled_waveform

    def _perform_resample(self, waveforms):
        """Resamples the waveform at the new frequency.

        This matches Kaldi's OfflineFeatureTpl ResampleWaveform which uses a
        LinearResample (resample a signal at linearly spaced intervals to
        up/downsample a signal). LinearResample (LR) means that the output
        signal is at linearly spaced intervals (i.e the output signal has a
        frequency of `new_freq`). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        https://ccrma.stanford.edu/~jos/resample/
        Theory_Ideal_Bandlimited_Interpolation.html

        https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56

        Arguments
        ---------
        waveforms : tensor
            The batch of audio signals to resample.

        Returns
        -------
        The waveforms at the new frequency.
        """

        # Compute output size and initialize
        batch_size, num_channels, wave_len = waveforms.shape
        window_size = self.weights.shape[1]
        tot_output_samp = self._output_samples(wave_len)
        resampled_waveform = paddle.zeros(
            (batch_size, num_channels, tot_output_samp),
        )
        self.weights = self.weights

        # Check weights are on correct device
        # if waveforms.device != self.weights.device:
        #     self.weights = self.weights.to(waveforms.device)

        # eye size: (num_channels, num_channels, 1)
        eye = paddle.eye(num_channels).unsqueeze(2)

        # Iterate over the phases in the polyphase filter
        for i in range(self.first_indices.shape[0]):
            wave_to_conv = waveforms
            first_index = int(self.first_indices[i].item())
            if first_index >= 0:
                # trim the signal as the filter will not be applied
                # before the first_index
                wave_to_conv = wave_to_conv[..., first_index:]

            # pad the right of the signal to allow partial convolutions
            # meaning compute values for partial windows (e.g. end of the
            # window is outside the signal length)
            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            wave_to_conv = paddle.nn.functional.pad(
                wave_to_conv, (left_padding, right_padding),
                data_format="NCL"
            )
            conv_wave = paddle.nn.functional.conv1d(
                x=wave_to_conv,
                weight=self.weights[i].tile([num_channels, 1, 1]),
                stride=self.conv_stride,
                groups=num_channels,
            )

            # we want conv_wave[:, i] to be at
            # output[:, i + n*conv_transpose_stride]
            # 这里不知道怎么实现？？？
            dilated_conv_wave = paddle.nn.functional.conv1d_transpose(
                conv_wave, eye, stride=self.conv_transpose_stride
            )

            # pad dilated_conv_wave so it reaches the output length if needed.
            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.shape[-1]
            right_padding = max(0, tot_output_samp - previous_padding)
            dilated_conv_wave = paddle.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding),
                data_format="NCL"
            )
            dilated_conv_wave = dilated_conv_wave[..., :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        """Based on LinearResample::GetNumOutputSamples.

        LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited
        interpolation to upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        Arguments
        ---------
        input_num_samp : int
            The number of samples in each example in the batch.

        Returns
        -------
        Number of samples in the output waveform.
        """

        # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
        # where tick_freq is the least common multiple of samp_in and
        # samp_out.
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
        """Based on LinearResample::SetIndexesAndWeights

        Retrieves the weights for resampling as well as the indices in which
        they are valid. LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a frequency
        of ``new_freq``). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        Returns
        -------
        - the place where each filter should start being applied
        - the filters to be applied to the signal for resampling
        """

        # Lowpass filter frequency depends on smaller of two frequencies
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        output_t = paddle.arange(
            start=0.0, end=self.output_samples
        )
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = paddle.ceil(min_t * self.orig_freq)
        max_input_index = paddle.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        j = paddle.arange(max_weight_width)
        input_index = min_input_index.unsqueeze(1) + j.unsqueeze(0)
        delta_t = (input_index / self.orig_freq) - output_t.unsqueeze(1)

        weights = paddle.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().less_than(paddle.to_tensor([window_width]))

        # raised-cosine (Hanning) window with width `window_width`
        weights[inside_window_indices] = 0.5 * (
            1
            + paddle.cos(
                2
                * math.pi
                * lowpass_cutoff
                / self.lowpass_filter_width
                * delta_t[inside_window_indices]
            )
        )

        t_eq_zero_indices = delta_t.equal(paddle.to_tensor([0.0]))
        t_not_eq_zero_indices = ~t_eq_zero_indices

        # sinc filter function
        weights[t_not_eq_zero_indices] *= paddle.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]
        ) / (math.pi * delta_t[t_not_eq_zero_indices])

        # limit of the function at t = 0
        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        # size (output_samples, max_weight_width)
        weights /= self.orig_freq

        self.first_indices = min_input_index
        self.weights = weights


class AddBabble(paddle.nn.Layer):
    """Simulate babble noise by mixing the signals in a batch.

    Arguments
    ---------
    speaker_count : int
        The number of signals to mix with the original signal.
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    mix_prob : float
        The probability that the batch of signals will be
        mixed with babble noise. By default, every signal is mixed.

    Example
    -------
    >>> import pytest
    >>> babbler = AddBabble()
    >>> dataset = ExtendedCSVDataset(
    ...     csvpath='samples/audio_samples/csv_example3.csv',
    ... )
    >>> loader = make_dataloader(dataset, batch_size=5)
    >>> speech, lengths = next(iter(loader)).at_position(0)
    >>> noisy = babbler(speech, lengths)
    """

    def __init__(
        self, speaker_count=3, snr_low=0, snr_high=0, mix_prob=1,
    ):
        super().__init__()
        self.speaker_count = speaker_count
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            A batch of audio signals to process, with shape `[batch, time]` or
            `[batch, time, channels]`.
        lengths : tensor
            The length of each audio in the batch, with shape `[batch]`.

        Returns
        -------
        Tensor with processed waveforms.
        """

        babbled_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)
        batch_size = len(waveforms)

        # Don't mix (return early) 1-`mix_prob` portion of the batches
        if paddle.rand([1]) > self.mix_prob:
            return babbled_waveform

        # Pick an SNR and use it to compute the mixture amplitude factors
        clean_amplitude = compute_amplitude(waveforms, lengths)
        SNR = paddle.rand([batch_size, 1])
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        babbled_waveform *= 1 - noise_amplitude_factor

        # For each speaker in the mixture, roll and add
        babble_waveform = waveforms.roll((1,), axis=0)
        babble_len = lengths.roll((1,), axis=0)
        for i in range(1, self.speaker_count):
            babble_waveform += waveforms.roll((1 + i,), axis=0)
            babble_len = paddle.max(babble_len, babble_len.roll((1,), axis=0))

        # Rescale and add to mixture
        babble_amplitude = compute_amplitude(babble_waveform, babble_len)
        babble_waveform *= new_noise_amplitude / (babble_amplitude + 1e-14)
        babbled_waveform += babble_waveform

        return babbled_waveform


class DropFreq(paddle.nn.Layer):
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
    >>> signal = read_audio('samples/audio_samples/example1.wav')
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
        if paddle.rand([1]) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(-1)

        # Pick number of frequencies to drop
        drop_count = paddle.randint(
            low=self.drop_count_low, high=self.drop_count_high + 1, shape=[1,],
        )
        # print("drop count: {}".format(drop_count))
        # Pick a frequency to drop
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequency = (
            paddle.rand(drop_count) * drop_range + self.drop_freq_low
        )

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = paddle.zeros([1, filter_length, 1],)
        drop_filter[0, pad, 0] = 1

        # Subtract each frequency
        # drop_frequency = paddle.to_tensor([0.6230, 0.2448, 0.6192], dtype="float32")
        # print("drop frequency: {}".format(drop_frequency))
        for frequency in drop_frequency:
            notch_kernel = notch_filter(
                frequency, filter_length, self.drop_width,
            )
            # print("notch kernel: {}".format(notch_kernel))
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)
            # print("drop filter: {}".format(drop_filter))
        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze(-1)


class DropChunk(paddle.nn.Layer):
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
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    >>> length = paddle.ones(1)
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
        lengths = (lengths * waveforms.shape[1]).astype("int64")
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
            shape=[batch_size,],
        )

        # Iterate batch to set mask
        for i in range(batch_size):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            length = paddle.randint(
                low=self.drop_length_low,
                high=self.drop_length_high + 1,
                shape=[drop_times[i],],
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
            start = paddle.randint(
                low=start_min, high=start_max + 1, shape=[drop_times[i],],
            )
            end = start + length

            # Update waveform
            if not self.noise_factor:
                for j in range(drop_times[i]):
                    # print("start: {}, end: {}".format(start[j].item(), end[j].item()))
                    if start[j] < end[j]:
                        dropped_waveform[i, start[j] : end[j]] = 0.0
            else:
                # Uniform distribution of -2 to +2 * avg amplitude should
                # preserve the average for normalization
                noise_max = 2 * clean_amplitude[i] * self.noise_factor
                for j in range(drop_times[i]):
                    # zero-center the noise distribution
                    noise_vec = paddle.rand([length[j]])
                    noise_vec = 2 * noise_max * noise_vec - noise_max
                    if start[j] < end[j]:
                        dropped_waveform[i, start[j] : end[j]] = noise_vec

        return dropped_waveform


class DoClip(paddle.nn.Layer):
    """This function mimics audio clipping by clamping the input tensor.

    Arguments
    ---------
    clip_low : float
        The low end of amplitudes for which to clip the signal.
    clip_high : float
        The high end of amplitudes for which to clip the signal.
    clip_prob : float
        The probability that the batch of signals will have a portion clipped.
        By default, every batch has portions clipped.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> clipper = DoClip(clip_low=0.01, clip_high=0.01)
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> clipped_signal = clipper(signal.unsqueeze(0))
    >>> "%.2f" % clipped_signal.max()
    '0.01'
    """

    def __init__(
        self, clip_low=0.5, clip_high=1, clip_prob=1,
    ):
        super().__init__()
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.clip_prob = clip_prob

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Don't clip (return early) 1-`clip_prob` portion of the batches
        if paddle.rand([1]) > self.clip_prob:
            return waveforms.clone()

        # Randomly select clip value
        clipping_range = self.clip_high - self.clip_low
        clip_value = paddle.rand([1],)[0] * clipping_range + self.clip_low

        # Apply clipping
        clipped_waveform = waveforms.clip(-clip_value, clip_value)

        return clipped_waveform
