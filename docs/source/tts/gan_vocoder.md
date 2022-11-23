# GAN Vocoders
This is a brief introduction of GAN Vocoders, we mainly introduce the losses of different vocoders here.

Model  | Generator Loss |Discriminator Loss
:-------------:| :------------:| :-----
Mel GAN | adversial loss <br> Feature Matching  | Multi-Scale Discriminator |
Parallel Wave GAN |adversial loss <br> Multi-resolution STFT loss  | adversial loss|
Multi-Band Mel GAN | adversial loss <br> full band Multi-resolution STFT loss <br> sub band Multi-resolution STFT loss |Multi-Scale Discriminator|
HiFi GAN |adversial loss <br> Feature Matching <br>  Mel-Spectrogram Loss | Multi-Scale Discriminator <br> Multi-Period Discriminator|
