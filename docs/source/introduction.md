# PaddleSpeech

## What is PaddleSpeech?
PaddleSpeech is an open-source toolkit on PaddlePaddle platform for two critical tasks in Speech -  Speech-To-Text (Automatic Speech Recognition, ASR) and Text-To-Speech Synthesis (TTS), with modules involving state-of-art and influential models.

## What can PaddleSpeech do?

### Speech-To-Text
(An introduce of ASR in PaddleSpeech is needed here!)

### Text-To-Speech
TTS mainly consists of components below:
- Implementation of models and commonly used neural network layers.
- Dataset abstraction and common data preprocessing pipelines.
- Ready-to-run experiments.

PaddleSpeech TTS provides you with a complete TTS pipeline, including:
- Text FrontEnd
    - Rule based Chinese frontend.
- Acoustic Models
    - FastSpeech2
    - SpeedySpeech
    - TransformerTTS
    - Tacotron2
- Vocoders
    - Multi Band MelGAN
    - Parallel WaveGAN
    - WaveFlow
- Voice Cloning
    - Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis
    - GE2E

Text-To-Speech  helps you to train TTS models with simple commands.
