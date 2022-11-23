# Features

### Dataset
* Aishell
* Librispeech
* THCHS30
* TIMIT

### Speech Recognition

* Non-Streaming
  * [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf)
  * [Transformer](https://arxiv.org/abs/1706.03762)
  * [Conformer](https://arxiv.org/abs/2005.08100)

* Streaming
  * [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf)
  * [U2](https://arxiv.org/pdf/2012.05481.pdf)

### Language Model

* Ngram

### Decoder

* ctc greedy
* ctc prefix beam search
* greedy
* beam search
* attention rescore

### Deployment

* Paddle Inference

### Aligment  

* MFA  
* CTC Alignment  

### Speech Frontend

* Audio
  * Auto Gain
* Feature
  * kaldi fbank
  * kaldi mfcc
  * linear
  * delta detla

### Speech Augmentation

* Audio
  - Volume Perturbation
  - Speed Perturbation
  - Shifting Perturbation
  - Online Bayesian normalization
  - Noise Perturbation
  - Impulse Response
* Spectrum
  - SpecAugment
  - Adaptive SpecAugment

### Tokenizer

* Chinese/English Character
* English Word
* Sentence Piece

### Word Segmentation

*  [mmseg](http://technology.chtsai.org/mmseg/)

### Grapheme To Phoneme

* syllable
* phoneme
