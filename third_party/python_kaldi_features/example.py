#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")

# note that generally nfilt=40 is used for speech recognition
fbank_feat = logfbank(sig,nfilt=23,lowfreq=20,dither=0,wintype='povey')

# the computed fbank coefficents of english.wav with dimension [110,23]
# [ 12.2865	12.6906	13.1765	15.714	16.064	15.7553	16.5746	16.9205	16.6472	16.1302	16.4576	16.7326	16.8864	17.7215	18.88	19.1377	19.1495	18.6683	18.3886	20.3506	20.2772	18.8248	18.1899
# 11.9198	13.146	14.7215	15.8642	17.4288	16.394	16.8238	16.1095	16.4297	16.6331	16.3163	16.5093	17.4981	18.3429	19.6555	19.6263	19.8435	19.0534	19.001	20.0287	19.7707	19.5852	19.1112
# ...
# ...
# the same with that using kaldi commands: compute-fbank-feats --dither=0.0


mfcc_feat = mfcc(sig,dither=0,useEnergy=True,wintype='povey')

# the computed mfcc coefficents of english.wav with dimension [110,13]
# [ 17.1337	-23.3651	-7.41751	-7.73686	-21.3682	-8.93884	-3.70843	4.68346	-16.0676	12.782	-7.24054	8.25089	10.7292
# 17.1692	-23.3028	-5.61872	-4.0075	-23.287	-20.6101	-5.51584	-6.15273	-14.4333	8.13052	-0.0345329	2.06274	-0.564298
# ...
# ...
# the same with that using kaldi commands: compute-mfcc-feats --dither=0.0

