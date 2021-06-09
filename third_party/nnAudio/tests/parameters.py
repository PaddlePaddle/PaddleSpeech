# Creating parameters for STFT test
"""
It is equivalent to 
[(1024, 128, 'ones'),
 (1024, 128, 'hann'),
 (1024, 128, 'hamming'),
 (2048, 128, 'ones'),
 (2048, 512, 'ones'),
 (2048, 128, 'hann'),
 (2048, 512, 'hann'),
 (2048, 128, 'hamming'),
 (2048, 512, 'hamming'),
 (None, None, None)]
"""

stft_parameters = []
n_fft = [1024,2048]
hop_length = {128,512,1024}
window = ['ones', 'hann', 'hamming']
for i in n_fft:
    for k in window:
        for j in hop_length:
            if j < (i/2):
                stft_parameters.append((i,j,k))
stft_parameters.append((256, None, 'hann'))

stft_with_win_parameters = []
n_fft = [512,1024]
win_length = [400, 900]
hop_length = {128,256}
for i in n_fft:
    for j in win_length:
        if j < i:     
            for k in hop_length:
                if k < (i/2):
                    stft_with_win_parameters.append((i,j,k))

mel_win_parameters = [(512,400), (1024, 1000)]