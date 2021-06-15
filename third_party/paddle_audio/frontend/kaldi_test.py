from typing import Tuple
import numpy as np
import paddle
import unittest

import decimal
import numpy
import math
import logging
from pathlib import Path

from third_party.paddle_audio.frontend import kaldi

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def do_dither(signal, dither_value=1.0):
    signal += numpy.random.normal(size=signal.shape) * dither_value
    return signal
    
def do_remove_dc_offset(signal):
    signal -= numpy.mean(signal)
    return signal

def do_preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append((1-coeff)*signal[0], signal[1:] - coeff * signal[:-1])


def framesig(sig, frame_len, frame_step, dither=1.0, preemph=0.97, remove_dc_offset=True, wintype='hamming', stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + (( slen - frame_len) // frame_step)

    # check kaldi/src/feat/feature-window.h
    padsignal = sig[:(numframes-1)*frame_step+frame_len]
    if wintype is 'povey':
        win = numpy.empty(frame_len)
        for i in range(frame_len):
            win[i] = (0.5-0.5*numpy.cos(2*numpy.pi/(frame_len-1)*i))**0.85     
    else: # the hamming window
        win = numpy.hamming(frame_len)
        
    if stride_trick:
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(win, (numframes, 1))
        
    frames = frames.astype(numpy.float32)
    raw_frames = numpy.zeros(frames.shape)
    for frm in range(frames.shape[0]):
        frames[frm,:] = do_dither(frames[frm,:], dither)        # dither
        frames[frm,:] = do_remove_dc_offset(frames[frm,:])      # remove dc offset
        raw_frames[frm,:] = frames[frm,:]
        frames[frm,:] = do_preemphasis(frames[frm,:], preemph)    # preemphasize

    return frames * win, raw_frames


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return numpy.square(magspec(frames, NFFT))


def framesig_without_dither_dc_preemphasize(sig, frame_len, frame_step, wintype='hamming', stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + (( slen - frame_len) // frame_step)

    # check kaldi/src/feat/feature-window.h
    padsignal = sig[:(numframes-1)*frame_step+frame_len]
    
    if wintype is 'povey':
        win = numpy.empty(frame_len)
        for i in range(frame_len):
            win[i] = (0.5-0.5*numpy.cos(2*numpy.pi/(frame_len-1)*i))**0.85 
    elif wintype == '':
        win = numpy.ones(frame_len)
    elif wintype == 'hann':
        win = numpy.hanning(frame_len)
    else: # the hamming window
        win = numpy.hamming(frame_len)
        
    if stride_trick:
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(win, (numframes, 1))
        
    frames = frames.astype(numpy.float32)
    raw_frames = frames
    return frames * win, raw_frames


def frames(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=40,nfft=512,lowfreq=0,highfreq=None, wintype='hamming'):
    frames_with_win, raw_frames = framesig_without_dither_dc_preemphasize(signal, winlen*samplerate, winstep*samplerate, wintype)
    return frames_with_win, raw_frames


def complexspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return complex_spec


def stft_with_window(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=40,nfft=512,lowfreq=0,highfreq=None,dither=1.0,remove_dc_offset=True, preemph=0.97, 
          wintype='hamming'):
    frames_with_win, raw_frames = framesig_without_dither_dc_preemphasize(signal, winlen*samplerate, winstep*samplerate, wintype)
    
    spec = magspec(frames_with_win, nfft) # nearly the same until this part
    scomplex = complexspec(frames_with_win, nfft)
    
    rspec = magspec(raw_frames, nfft)
    rcomplex = complexspec(raw_frames, nfft)
    return spec, scomplex, rspec, rcomplex


class TestKaldiFE(unittest.TestCase):
    def setUp(self):
        self. this_dir = Path(__file__).parent
        
        self.wavpath = str(self.this_dir / 'english.wav')
        self.winlen=0.025 # ms
        self.winstep=0.01 # ms
        self.nfft=512
        self.lowfreq = 0
        self.highfreq = None
        self.wintype='hamm'
        self.nfilt=40
        
        
    def test_read(self):
        import scipy.io.wavfile as wav
        rate, sig = wav.read(self.wavpath)
        sr, wav = kaldi.read(self.wavpath)
        self.assertTrue(np.all(sig == wav))
        self.assertEqual(rate, sr)
        
    def test_frames(self):
        sr, wav = kaldi.read(self.wavpath)
        _, fs = frames(wav, samplerate=sr, 
                            winlen=self.winlen, winstep=self.winstep, 
                            nfilt=self.nfilt, nfft=self.nfft, 
                            lowfreq=self.lowfreq, highfreq=self.highfreq, 
                            wintype=self.wintype)
        
        t_wav = paddle.to_tensor([wav], dtype='float32')
        t_wavlen = paddle.to_tensor([len(wav)])
        t_fs, t_nframe = kaldi.frames(t_wav, t_wavlen, sr, self.winlen, self.winstep, clip=False)
        t_fs = t_fs.astype(fs.dtype)[0]
        
        self.assertEqual(t_nframe.item(), fs.shape[0])
        self.assertTrue(np.allclose(t_fs.numpy(), fs))
        
        
    def test_stft(self):
        sr, wav = kaldi.read(self.wavpath)
        
        for wintype in ['', 'hamm', 'hann', 'povey']:
            print(wintype)
            self.wintype=wintype
            _, stft_c_win, _, _ = stft_with_window(wav, samplerate=sr, 
                                winlen=self.winlen, winstep=self.winstep, 
                                nfilt=self.nfilt, nfft=self.nfft, 
                                lowfreq=self.lowfreq, highfreq=self.highfreq, 
                                wintype=self.wintype)
            print('py', stft_c_win.real)
            print('py', stft_c_win.imag)
            
            t_wav = paddle.to_tensor([wav], dtype='float32')
            t_wavlen = paddle.to_tensor([len(wav)])
            
            stft_class = kaldi.STFT(self.nfft, sr, self.winlen, self.winstep, window_type=self.wintype, clip=False)
            t_stft, t_nframe = stft_class(t_wav, t_wavlen)
            t_stft = t_stft.astype(stft_c_win.real.dtype)[0]
            t_real = t_stft[:, :, 0]
            t_imag = t_stft[:, :, 1]
            print('pd', t_real.numpy())
            print('pd', t_imag.numpy())
            
            self.assertEqual(t_nframe.item(), stft_c_win.real.shape[0])

            self.assertLess(np.sum(t_real.numpy()) - np.sum(stft_c_win.real), 1)
            print(np.sum(t_real.numpy()))
            print(np.sum(stft_c_win.real))
            self.assertTrue(np.allclose(t_real.numpy(), stft_c_win.real, atol=1e-1))
            
            self.assertLess(np.sum(t_imag.numpy()) - np.sum(stft_c_win.imag), 1)
            print(np.sum(t_imag.numpy()))
            print(np.sum(stft_c_win.imag))
            self.assertTrue(np.allclose(t_imag.numpy(), stft_c_win.imag, atol=1e-1))
        
    def test_magspec(self):
        sr, wav = kaldi.read(self.wavpath)
        
        for wintype in ['', 'hamm', 'hann', 'povey']:
            print(wintype)
            self.wintype=wintype
            stft_win, _, _, _ = stft_with_window(wav, samplerate=sr, 
                                winlen=self.winlen, winstep=self.winstep, 
                                nfilt=self.nfilt, nfft=self.nfft, 
                                lowfreq=self.lowfreq, highfreq=self.highfreq, 
                                wintype=self.wintype)
            print('py', stft_win)

            t_wav = paddle.to_tensor([wav], dtype='float32')
            t_wavlen = paddle.to_tensor([len(wav)])
            
            stft_class = kaldi.STFT(self.nfft, sr, self.winlen, self.winstep, window_type=self.wintype, clip=False)
            t_stft, t_nframe = stft_class(t_wav, t_wavlen)
            t_stft = t_stft.astype(stft_win.dtype)
            t_spec = kaldi.magspec(t_stft)[0]
            print('pd', t_spec.numpy())

            self.assertEqual(t_nframe.item(), stft_win.shape[0])
            
            self.assertLess(np.sum(t_spec.numpy()) - np.sum(stft_win), 1)
            print(np.sum(t_spec.numpy()))
            print(np.sum(stft_win))
            self.assertTrue(np.allclose(t_spec.numpy(), stft_win, atol=1e-1))
            
    def test_powspec(self):
        sr, wav = kaldi.read(self.wavpath)
        
        for wintype in ['', 'hamm', 'hann', 'povey']:
            print(wintype)
            self.wintype=wintype
            stft_win, _, _, _ = stft_with_window(wav, samplerate=sr, 
                                winlen=self.winlen, winstep=self.winstep, 
                                nfilt=self.nfilt, nfft=self.nfft, 
                                lowfreq=self.lowfreq, highfreq=self.highfreq, 
                                wintype=self.wintype)
            stft_win = np.square(stft_win)
            print('py', stft_win)

            t_wav = paddle.to_tensor([wav], dtype='float32')
            t_wavlen = paddle.to_tensor([len(wav)])
            
            stft_class = kaldi.STFT(self.nfft, sr, self.winlen, self.winstep, window_type=self.wintype, clip=False)
            t_stft, t_nframe = stft_class(t_wav, t_wavlen)
            t_stft = t_stft.astype(stft_win.dtype)
            t_spec = kaldi.powspec(t_stft)[0]
            print('pd', t_spec.numpy())

            self.assertEqual(t_nframe.item(), stft_win.shape[0])
            
            self.assertLess(np.sum(t_spec.numpy() - stft_win), 2e4)
            print(np.sum(t_spec.numpy()))
            print(np.sum(stft_win))
            self.assertTrue(np.allclose(t_spec.numpy(), stft_win, atol=1e2))


# from python_speech_features import mfcc
# from python_speech_features import delta
# from python_speech_features import logfbank
# import scipy.io.wavfile as wav

# (rate,sig) = wav.read("english.wav")

# # note that generally nfilt=40 is used for speech recognition
# fbank_feat = logfbank(sig,nfilt=23,lowfreq=20,dither=0,wintype='povey')

# # the computed fbank coefficents of english.wav with dimension [110,23]
# # [ 12.2865	12.6906	13.1765	15.714	16.064	15.7553	16.5746	16.9205	16.6472	16.1302	16.4576	16.7326	16.8864	17.7215	18.88	19.1377	19.1495	18.6683	18.3886	20.3506	20.2772	18.8248	18.1899
# # 11.9198	13.146	14.7215	15.8642	17.4288	16.394	16.8238	16.1095	16.4297	16.6331	16.3163	16.5093	17.4981	18.3429	19.6555	19.6263	19.8435	19.0534	19.001	20.0287	19.7707	19.5852	19.1112
# # ...
# # ...
# # the same with that using kaldi commands: compute-fbank-feats --dither=0.0


# mfcc_feat = mfcc(sig,dither=0,useEnergy=True,wintype='povey')

# # the computed mfcc coefficents of english.wav with dimension [110,13]
# # [ 17.1337	-23.3651	-7.41751	-7.73686	-21.3682	-8.93884	-3.70843	4.68346	-16.0676	12.782	-7.24054	8.25089	10.7292
# # 17.1692	-23.3028	-5.61872	-4.0075	-23.287	-20.6101	-5.51584	-6.15273	-14.4333	8.13052	-0.0345329	2.06274	-0.564298
# # ...
# # ...
# # the same with that using kaldi commands: compute-mfcc-feats --dither=0.0



if __name__ == '__main__':
    unittest.main()