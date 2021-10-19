# Parakeet - PAddle PARAllel text-to-speech toolKIT

## What is Parakeet?
Parakeet is a deep learning based text-to-speech toolkit built upon paddlepaddle framework. It aims to provide a flexible, efficient and state-of-the-art text-to-speech toolkit for the open-source community. It includes many influential TTS models proposed by Baidu Research and other research groups.

## What can Parakeet do?
Parakeet mainly consists of components below:
- Implementation of models and commonly used neural network layers.
- Dataset abstraction and common data preprocessing pipelines.
- Ready-to-run experiments.

Parakeet provides you with a complete TTS pipeline, including:
- Text FrontEnd
    - Rule based Chinese frontend.
- Acoustic Models
    - FastSpeech2
    - SpeedySpeech
    - TransformerTTS
    - Tacotron2
- Vocoders
    - Parallel WaveGAN
    - WaveFlow
- Voice Cloning
    - Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis
    - GE2E

Parakeet helps you to train TTS models with simple commands.
