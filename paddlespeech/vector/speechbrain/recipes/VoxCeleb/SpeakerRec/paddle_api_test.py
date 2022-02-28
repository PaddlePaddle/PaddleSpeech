import paddle
import numpy as np
paddle.device.set_device("cpu")

x = np.exp(3j * np.pi * np.arange(7) / 7)
xp = paddle.to_tensor(x)
fft_xp = paddle.fft.fft(xp).unsqueeze(0)
print(fft_xp)
fft_real = fft_xp.real()
fft_imag = fft_xp.imag()
print(fft_real)
print(fft_imag)
fft_real = fft_real.pow(2).unsqueeze(-2)
fft_imag = fft_imag.pow(2).unsqueeze(-2)
fft_xp = paddle.concat([fft_real, fft_imag], axis=-2).transpose(perm=[0, 2, 1]).sum(-1)
# print(fft_real)
# print(fft_imag)

# fft_xp = paddle.concat([fft_real, fft_imag], axis=-1)
print(fft_xp)