# PaddleSpeech

## What is PaddleSpeech?
PaddleSpeech is an open-source toolkit on the PaddlePaddle platform for two critical tasks in Speech -  Speech-to-Text (Automatic Speech Recognition, ASR) and Text-to-Speech Synthesis (TTS), with modules involving state-of-art and influential models.

## What can PaddleSpeech do?

### Speech-to-Text
PaddleSpeech ASR mainly consists of components below:
- Implementation of models and commonly used neural network layers.
- Dataset abstraction and common data preprocessing pipelines.
- Ready-to-run experiments.

PaddleSpeech ASR provides you with a complete ASR pipeline, including:
- Data Preparation
    - Build vocabulary
    - Compute Cepstral mean and variance normalization (CMVN)
    - Featrue extraction
        - linear
        - fbank (also support kaldi feature)
        - mfcc
- Acoustic Models
    - Deepspeech2 (Streaming and Non-Streaming)
    - Transformer (Streaming and Non-Streaming)
    - Conformer (Streaming and Non-Streaming)
- Decoder
    - ctc greedy search (used in DeepSpeech2, Transformer and Conformer)
    - ctc beam search (used in DeepSpeech2, Transformer and Conformer)
    - attention decoding (used in Transformer and Conformer)
    - attention rescoring (used in Transformer and Conformer)

Speech-to-Text helps you train the ASR model very simply.

### Text-to-Speech
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
    - Transfer Learning from Speaker Verification to Multispeaker Text-to-Speech Synthesis
    - GE2E

Text-to-Speech helps you to train TTS models with simple commands.
