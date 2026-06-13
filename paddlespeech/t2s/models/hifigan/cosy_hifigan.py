import paddle

"""HIFI-GAN"""
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import get_window
from paddlespeech.t2s.modules.transformer.activation import Snake
from paddlespeech.t2s.models.CosyVoice.common import get_padding, init_weights

"""hifigan based generator implementation.

This code is modified from https://github.com/jik876/hifi-gan
 ,https://github.com/kan-bayashi/ParallelWaveGAN and
 https://github.com/NVIDIA/BigVGAN

"""


class ResBlock(paddle.nn.Layer):
    """Residual block module in HiFiGAN/BigVGAN."""

    def __init__(self, channels: int=512, kernel_size: int=3, dilations:
        List[int]=[1, 3, 5]):
        super(ResBlock, self).__init__()
        self.convs1 = paddle.nn.LayerList()
        self.convs2 = paddle.nn.LayerList()
        for dilation in dilations:
            self.convs1.append(paddle.nn.Conv1D(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation)))
            self.convs2.append(paddle.nn.Conv1D(channels, channels, kernel_size, 1, dilation=1,padding=get_padding(kernel_size, 1)))
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = paddle.nn.LayerList(sublayers=[Snake(channels,
            alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = paddle.nn.LayerList(sublayers=[Snake(channels,
            alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            paddle.nn.utils.remove_weight_norm(layer=self.convs1[idx])
            paddle.nn.utils.remove_weight_norm(layer=self.convs2[idx])


class SineGen(paddle.nn.Layer):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=
        0.003, voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).astype(paddle.float32)
        return uv

    @paddle.no_grad()
    def forward(self, f0):
        """
        :param f0: [B, 1, sample_len], Hz
        :return: [B, 1, sample_len]
        """
        F_mat = paddle.zeros([f0.size(0), self.harmonic_num + 1, f0.size(-1)]).to(f0.place)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i:i + 1, :] = f0 * (i + 1) / self.sampling_rate
        theta_mat = 2 * np.pi * (paddle.cumsum(F_mat, axis=-1) % 1)
        u_dist = paddle.distribution.Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(shape=(f0.size(0), self.harmonic_num + 1, 1)
            ).to(F_mat.place)
        phase_vec[:, 0, :] = 0
        sine_waves = self.sine_amp * paddle.sin(theta_mat + phase_vec)
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * paddle.randn(shape=sine_waves.shape, dtype=
            sine_waves.dtype)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

class SourceModuleHnNSF(paddle.nn.Layer):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0, sinegen_type='1', causal=False):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        if sinegen_type == '1':
            self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        else:
            self.l_sin_gen = SineGen2(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod, causal=causal)
        self.l_linear = paddle.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = paddle.nn.Tanh()
        self.causal = causal
        paddle.seed(1986)
        
        if causal is True:
            self.uv = paddle.rand(shape=[1, 300 * 24000, 1])
            self.register_buffer('uv_buffer', self.uv)

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        with paddle.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        
        if not self.training and self.causal:
            noise = self.uv_buffer[:, :uv.shape[1]] * self.sine_amp / 3
        else:
            noise = paddle.randn(shape=uv.shape, dtype=uv.dtype) * self.sine_amp / 3
        return sine_merge, noise, uv
# class SourceModuleHnNSF(paddle.nn.Layer):
#     """ SourceModule for hn-nsf
#     SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
#                  add_noise_std=0.003, voiced_threshod=0)
#     sampling_rate: sampling_rate in Hz
#     harmonic_num: number of harmonic above F0 (default: 0)
#     sine_amp: amplitude of sine source signal (default: 0.1)
#     add_noise_std: std of additive Gaussian noise (default: 0.003)
#         note that amplitude of noise in unvoiced is decided
#         by sine_amp
#     voiced_threshold: threhold to set U/V given F0 (default: 0)
#     Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
#     F0_sampled (batchsize, length, 1)
#     Sine_source (batchsize, length, 1)
#     noise_source (batchsize, length 1)
#     uv (batchsize, length, 1)
#     """

#     def __init__(self, sampling_rate, upsample_scale, harmonic_num=0,
#         sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
#         super(SourceModuleHnNSF, self).__init__()
#         self.sine_amp = sine_amp
#         self.noise_std = add_noise_std
#         self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp,
#             add_noise_std, voiced_threshod)
#         self.l_linear = paddle.nn.Linear(in_features=harmonic_num + 1,
#             out_features=1)
#         self.l_tanh = paddle.nn.Tanh()

#     def forward(self, x):
#         """
#         Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
#         F0_sampled (batchsize, length, 1)
#         Sine_source (batchsize, length, 1)
#         noise_source (batchsize, length 1)
#         """
#         with paddle.no_grad():
#             sine_wavs, uv, _ = self.l_sin_gen(paddle.transpose(x,perm=[0,2,1]))
#             sine_wavs = paddle.transpose(sine_wavs,perm=[0,2,1])
#             uv = paddle.transpose(uv,perm=[0,2,1])
#         sine_merge = self.l_tanh(self.l_linear(sine_wavs))

#         noise = paddle.randn(shape=uv.shape, dtype=uv.dtype
#             ) * self.sine_amp / 3
#         return sine_merge, noise, uv


# class SineGen2(paddle.nn.Layer):
#     """ Definition of sine generator
#     SineGen(samp_rate, harmonic_num = 0,
#             sine_amp = 0.1, noise_std = 0.003,
#             voiced_threshold = 0,
#             flag_for_pulse=False)
#     samp_rate: sampling rate in Hz
#     harmonic_num: number of harmonic overtones (default 0)
#     sine_amp: amplitude of sine-wavefrom (default 0.1)
#     noise_std: std of Gaussian noise (default 0.003)
#     voiced_thoreshold: F0 threshold for U/V classification (default 0)
#     flag_for_pulse: this SinGen is used inside PulseGen (default False)
#     Note: when flag_for_pulse is True, the first time step of a voiced
#         segment is always sin(np.pi) or cos(0)
#     """

#     def __init__(self, samp_rate, upsample_scale, harmonic_num=0, sine_amp=
#         0.1, noise_std=0.003, voiced_threshold=0, flag_for_pulse=False):
#         super(SineGen2, self).__init__()
#         self.sine_amp = sine_amp
#         self.noise_std = noise_std
#         self.harmonic_num = harmonic_num
#         self.axis = self.harmonic_num + 1
#         self.sampling_rate = samp_rate
#         self.voiced_threshold = voiced_threshold
#         self.flag_for_pulse = flag_for_pulse
#         self.upsample_scale = upsample_scale

#     def _f02uv(self, f0):
#         uv = (f0 > self.voiced_threshold).astype(paddle.float32)
#         return uv

#     def _f02sine(self, f0_values):
#         """ f0_values: (batchsize, length, axis)
#             where axis indicates fundamental tone and overtones
#         """
#         rad_values = f0_values / self.sampling_rate % 1
#         rand_ini = paddle.rand(shape=[f0_values.shape[0], f0_values.shape[2]])
#         rand_ini[:, 0] = 0
#         rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
#         if not self.flag_for_pulse:
#             x = paddle.transpose(rad_values,perm = [0,2,1])
            
#             rad_values = paddle.transpose(paddle.nn.functional.interpolate(x=x, scale_factor=1 / self.upsample_scale, mode='linear'),perm = [0,2,1])
#             phase = paddle.cumsum(rad_values, axis=1) * 2 * np.pi
#             phase = paddle.transpose(paddle.nn.functional.interpolate(x=paddle.transpose(phase,perm = [0,2,1]) * self.upsample_scale, scale_factor=int(self.upsample_scale),mode='linear'),perm = [0,2,1])
#             sines = paddle.sin(phase)
#         else:
#             uv = self._f02uv(f0_values)
#             uv_1 = paddle.roll(uv, shifts=-1, axis=1)
#             uv_1[:, -1, :] = 1
#             u_loc = (uv < 1) * (uv_1 > 0)
#             tmp_cumsum = paddle.cumsum(rad_values, axis=1)
#             for idx in range(f0_values.shape[0]):
#                 temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
#                 temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
#                 tmp_cumsum[idx, :, :] = 0
#                 tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
#             i_phase = paddle.cumsum(rad_values - tmp_cumsum, axis=1)
#             sines = paddle.cos(i_phase * 2 * np.pi)
#         return sines

#     def forward(self, f0):
#         """ sine_tensor, uv = forward(f0)
#         input F0: tensor(batchsize=1, length, axis=1)
#                   f0 for unvoiced steps should be 0
#         output sine_tensor: tensor(batchsize=1, length, axis)
#         output uv: tensor(batchsize=1, length, 1)
#         """
#         paddle.seed(1986)
#         fn = paddle.multiply(f0, paddle.to_tensor([[range(1, self.harmonic_num +
#             2)]],dtype='float32',place=f0.place))

#         sine_waves = self._f02sine(fn) * self.sine_amp
#         uv = self._f02uv(f0)
#         noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
#         noise = noise_amp * paddle.randn(shape=sine_waves.shape, dtype=
#             sine_waves.dtype)
#         sine_waves = sine_waves * uv + noise

#         return sine_waves, uv, noise
class SineGen2(paddle.nn.Layer):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False,
                 causal=False):
        super(SineGen2, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale
        self.causal = causal
        paddle.seed(1986)
        if causal is True:
            self.rand_ini = paddle.rand(shape=[1, 9])
            self.rand_ini[:, 0] = 0
            self.sine_waves = paddle.rand(shape=[1, 300 * 24000, 9])
            self.register_buffer('rand_ini_buffer', self.rand_ini)
            self.register_buffer('sine_waves_buffer', self.sine_waves)

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).astype('float32')
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim) """
        rad_values = (f0_values / self.sampling_rate) % 1
        if not self.training and self.causal:
            rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini_buffer
        else:
            rand_ini = paddle.rand(shape=[f0_values.shape[0], f0_values.shape[2]])
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        
        if not self.flag_for_pulse:
            scale_factor_down = 1.0 / self.upsample_scale 
            
            rad_values = paddle.nn.functional.interpolate(
                rad_values.transpose([0, 2, 1]),
                scale_factor=scale_factor_down,
                mode="linear"
            ).transpose([0, 2, 1])
            phase = paddle.cumsum(rad_values, axis=1) * 2 * np.pi
            
            interpolate_mode = "nearest" if self.causal else 'linear'
            phase = paddle.transpose(paddle.nn.functional.interpolate(
                paddle.transpose(phase,perm=[0, 2, 1])*self.upsample_scale,
                scale_factor=float(self.upsample_scale), 
                mode=interpolate_mode
            ),perm = [0, 2, 1])

            sines = paddle.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            
            uv_1 = paddle.roll(uv, shifts=-1, axis=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            tmp_cumsum = paddle.cumsum(rad_values, axis=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            i_phase = paddle.cumsum(rad_values - tmp_cumsum, axis=1)
            sines = paddle.cos(i_phase * 2 * np.pi)
        
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0) """
        paddle.seed(1986)
        harmonic_coeffs = paddle.to_tensor(
            [list(range(1, self.harmonic_num + 2))], 
            dtype='float32'
        ).reshape([1, 1, -1])
        
        fn = f0 * harmonic_coeffs
        
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        
        if not self.training and self.causal:
            noise = noise_amp * self.sine_waves_buffer[:, :sine_waves.shape[1]]
        else:
            noise = noise_amp * paddle.randn(
                shape=sine_waves.shape, 
                dtype=sine_waves.dtype
            )
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

class SourceModuleHnNSF2(paddle.nn.Layer):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0,
        sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF2, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen2(sampling_rate, upsample_scale,
            harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = paddle.nn.Linear(in_features=harmonic_num + 1,
            out_features=1)
        self.l_tanh = paddle.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        paddle.seed(1986)
        with paddle.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = paddle.randn(shape=uv.shape, dtype=uv.dtype
            ) * self.sine_amp / 3
        return sine_merge, noise, uv


class HiFTGenerator(paddle.nn.Layer):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """

    def __init__(self, in_channels: int=80, base_channels: int=512,
        nb_harmonics: int=8, sampling_rate: int=22050, nsf_alpha: float=0.1,
        nsf_sigma: float=0.003, nsf_voiced_threshold: float=10,
        upsample_rates: List[int]=[8, 8], upsample_kernel_sizes: List[int]=
        [16, 16], istft_params: Dict[str, int]={'n_fft': 16, 'hop_len': 4},
        resblock_kernel_sizes: List[int]=[3, 7, 11],
        resblock_dilation_sizes: List[List[int]]=[[1, 3, 5], [1, 3, 5], [1,
        3, 5]], source_resblock_kernel_sizes: List[int]=[7, 11],
        source_resblock_dilation_sizes: List[List[int]]=[[1, 3, 5], [1, 3, 
        5]], lrelu_slope: float=0.1, audio_limit: float=0.99, f0_predictor:
        paddle.nn.Layer=None):
        super(HiFTGenerator, self).__init__()
        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
            sinegen_type='1' if self.sampling_rate == 22050 else '2',
            causal=False)
        self.f0_upsamp = paddle.nn.Upsample(scale_factor=(1,int(np.prod( upsample_rates) * istft_params['hop_len'])))
        self.conv_pre = paddle.nn.Conv1D(in_channels = in_channels, out_channels = base_channels, kernel_size=7, stride=1, padding=3)
        self.ups = paddle.nn.LayerList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(paddle.nn.Conv1DTranspose(in_channels=base_channels // 2 ** i,out_channels=base_channels // 2 ** (i + 1), kernel_size=k,stride=u, padding=(k - u) // 2))
        self.source_downs = paddle.nn.LayerList()
        self.source_resblocks = paddle.nn.LayerList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1],
            source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(paddle.nn.Conv1D(istft_params[
                    'n_fft'] + 2, base_channels // 2 ** (i + 1), 1, 1))
            else:
                self.source_downs.append(paddle.nn.Conv1D(
                    in_channels=istft_params['n_fft'] + 2,
                    out_channels=base_channels // (2 ** (i + 1)),
                    kernel_size=(u * 2,),
                    stride=(u,),
                    padding=int(u // 2),
                ))
            self.source_resblocks.append(ResBlock(base_channels // 2 ** (i +
                1), k, d))
        self.resblocks = paddle.nn.LayerList()
        for i in range(len(self.ups)):
            ch = base_channels // 2 ** (i + 1)
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes,
                resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = paddle.nn.Conv1D(ch, istft_params['n_fft'] + 2, 7, 1, padding=3)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = paddle.nn.Pad1D(padding=(1, 0), mode='reflect')
        self.stft_window = paddle.to_tensor(get_window('hann',
            istft_params['n_fft'], fftbins=True).astype(np.float32))
        self.f0_predictor = f0_predictor

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            paddle.nn.utils.remove_weight_norm(layer=l)
        for l in self.resblocks:
            l.remove_weight_norm()
        paddle.nn.utils.remove_weight_norm(layer=self.conv_pre)
        paddle.nn.utils.remove_weight_norm(layer=self.conv_post)
        self.m_source.remove_weight_norm()
        for l in self.source_downs:
            paddle.nn.utils.remove_weight_norm(layer=l)
        for l in self.source_resblocks:
            l.remove_weight_norm()

    def _stft(self, x):
        spec = paddle.signal.stft(x=x, n_fft=self.istft_params['n_fft'],
            hop_length=self.istft_params['hop_len'], win_length=self.
            istft_params['n_fft'], window=self.stft_window.to(x.place))
        spec = paddle.as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = paddle.clip(magnitude, max=100.0)
        real = magnitude * paddle.cos(phase)
        img = magnitude * paddle.sin(phase)
        inverse_transform = paddle.signal.istft(x=paddle.complex(real, img),
            n_fft=self.istft_params['n_fft'], hop_length=self.istft_params[
            'hop_len'], win_length=self.istft_params['n_fft'], window=self.
            stft_window.to(magnitude.place))
        return inverse_transform

    def decode(self, x: paddle.Tensor, s: paddle.Tensor=paddle.zeros([1, 1, 0])
        ) ->paddle.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = paddle.cat([s_stft_real, s_stft_imag], dim=1)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=self.
                lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = paddle.nn.functional.leaky_relu(x=x)
        x = self.conv_post(x)
        magnitude = paddle.exp(x=x[:, :self.istft_params['n_fft'] // 2 + 1, :])
        phase = paddle.sin(x[:, self.istft_params['n_fft'] // 2 + 1:, :])
        x = self._istft(magnitude, phase)
        x = paddle.clip(x, -self.audio_limit, self.audio_limit)
        return x

    def forward(self, batch: dict) ->Dict[str,
        Optional[paddle.Tensor]]:
        speech_feat = paddle.transpose(batch['speech_feat'],perm = [0,2,1]).to(device)
        f0 = self.f0_predictor(speech_feat)
        s = paddle.transpose(self.f0_upsamp(f0[:, None]),perm = [0,2,1])
        s, _, _ = self.m_source(s)
        s = paddle.transpose(s,perm = [0,2,1])
        
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, f0

    @paddle.no_grad()
    def inference(self, speech_feat: paddle.Tensor, cache_source: paddle.
        Tensor=paddle.zeros([1, 1, 0])) ->paddle.Tensor:
        paddle.seed(1986)
        f0 = self.f0_predictor(speech_feat)
        f0_4d = f0[:, None].unsqueeze(2)
        s_4d = self.f0_upsamp(f0_4d)
        s_3d = s_4d.squeeze(2)     
        s = paddle.transpose(s_3d, perm=[0, 2, 1]) 
        
        s, _, _ = self.m_source(s)
        s = paddle.transpose(s,perm = [0,2,1])
        
        if cache_source.shape[2] != 0:
            s[:, :, :cache_source.shape[2]] = cache_source
        generated_speech = self.decode(x=speech_feat, s=s)
        
        return generated_speech, s
