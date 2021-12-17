# Models introduction
TTS system mainly includes three modules: `Text Frontend`, `Acoustic model` and `Vocoder`. We introduce a rule-based Chinese text frontend in [cn_text_frontend.md](./cn_text_frontend.md). Here, we will introduce acoustic models and vocoders, which are trainable.

The main processes of TTS include:
1. Convert the original text into characters/phonemes, through the `text frontend` module.
2. Convert characters/phonemes into acoustic features, such as linear spectrogram, mel spectrogram, LPC features, etc. through `Acoustic models`.
3. Convert acoustic features into waveforms through `Vocoders`.

A simple text frontend module can be implemented by rules. Acoustic models and vocoders need to be trained. The models provided by PaddleSpeech TTS are acoustic models and vocoders.

## Acoustic Models
### Modeling Objectives of Acoustic Models
Modeling the mapping relationship between text sequences and speech features：
```text
text X = {x1,...,xM}
specch Y = {y1,...yN}
```
Modeling Objectives:
```text
Ω = argmax p(Y|X,Ω)
```
### Modeling process of Acoustic Models
At present, there are two mainstream acoustic model structures.

- Frame level acoustic model:
   - Duration model (M Tokens - > N Frames).
   - Acoustic decoder (N Frames - > N Frames).

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/frame_level_am.png" width=500 /> <br>
</div>

- Sequence to sequence acoustic model:
    - M Tokens - > N Frames.

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/seq2seq_am.png" width=500 /> <br>
</div>

### Tacotron2
 [Tacotron](https://arxiv.org/abs/1703.10135)  is the first end-to-end acoustic model based on deep learning, and it is also the most widely used acoustic model.

[Tacotron2](https://arxiv.org/abs/1712.05884) is the Improvement of Tacotron.
#### Tacotron
**Features of Tacotron:**
- Encoder.
   - CBHG.
   - Input: character sequence.
- Decoder.
    - Global soft attention.
    - unidirectional RNN.
    - Autoregressive teacher force training (input real speech feature).
    - Multi frame prediction.
    - CBHG postprocess.
    - Vocoder: Griffin-Lim.
<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/tacotron.png" width=700 /> <br>
</div>

**Advantage of Tacotron:**
- No need for complex text frontend analysis modules.
- No need for an additional duration model.
- Greatly simplifies the acoustic model construction process and reduces the dependence of speech synthesis tasks on domain knowledge.

**Disadvantages of Tacotron:**
- The CBHG  is complex and the amount of parameters is relatively large.
- Global soft attention.
- Poor stability for speech synthesis tasks.
- In training, the less the number of speech frames predicted at each moment, the more difficult it is to train.
-  Phase problem in Griffin-Lim causes speech distortion during wave reconstruction.
- The autoregressive decoder cannot be stopped during the generation process.

#### Tacotron2
**Features of Tacotron2:**
- Reduction of parameters.
   - CBHG -> PostNet (3 Conv layers + BLSTM or 5 Conv layers).
   - remove Attention RNN.
- Speech distortion caused by Griffin-Lim.
    - WaveNet.
- Improvements of PostNet.
   - CBHG -> 5 Conv layers.
   -  The input and output of the PostNet calculate `L2` loss with real Mel spectrogram.
   - Residual connection.
- Bad stop in an autoregressive decoder.
   - Predict whether it should stop at each moment of decoding (stop token).
   - Set a threshold to determine whether to stop generating when decoding.
- Stability of attention.
   - Location-aware attention.
   - The alignment matrix of the previous time is considered at step `t` of the decoder.

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/tacotron2.png" width=500 /> <br>
</div>

You can find PaddleSpeech TTS's tacotron2 with LJSpeech dataset example at [examples/ljspeech/tts0](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/tts0).

### TransformerTTS
**Disadvantages of the Tacotrons:**
- Encoder and decoder are relatively weak at global information modeling
   - Vanishing gradient of RNN.
   - Fixed-length context modeling problem in CNN kernel.
- Training is relatively inefficient.
- The attention is not robust enough and the stability is poor.

Transformer TTS is a combination of Tacotron2 and Transformer.

#### Transformer
 [Transformer](https://arxiv.org/abs/1706.03762) is a seq2seq model based entirely on an attention mechanism.

**Features of Transformer:**
- Encoder.
    - `N` blocks based on self-attention mechanism.
    - Positional Encoding.
- Decoder.
    - `N` blocks based on self-attention mechanism.
    - Add Mask to the self-attention in blocks to cover up the information after the `t` step.
    - Attentions between encoder and decoder.
    - Positional Encoding.

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/transformer.png" width=500 /> <br>
</div>

#### Transformer TTS
Transformer TTS is a seq2seq acoustic model based on Transformer and Tacotron2.

**Motivations：**
- RNNs in Tacotron2  make the inefficiency of training.
- Vanishing gradient of RNN makes the model's ability to model long-term contexts weak.
- Self-attention doesn't contain any recursive structure which can be trained in parallel.
- Self-attention can model global context information well.

**Features of Transformer TTS:**
- Add conv based PreNet in encoder and decoder.
- Stop Token in decoder controls when to stop autoregressive generation.
- Add PostNet after decoder to improve the quality of synthetic speech.
- Scaled position encoding.
    - Uniform scale position encoding may have a negative impact on input or output sequences.

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/transformer_tts.png" width=500 /> <br>
</div>

**Disadvantages of Transformer TTS:**
- The ability of position encoding for timing information is still relatively weak.
- The ability to perceive local information is weak, and local information is more related to pronunciation.
- Stability is worse than Tacotron2.

You can find PaddleSpeech TTS's Transformer TTS with LJSpeech dataset example at [examples/ljspeech/tts1](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/tts1).


### FastSpeech2
**Disadvantage of seq2seq models:**
- In the seq2seq model based on attention, no matter how to improve the attention mechanism, it's difficult to avoid generation errors in the decoding stage.

Frame-level acoustic models use duration models to determine the pronunciation duration of phonemes, and the frame-level mapping does not have the uncertainty of sequence generation.

In seq2saq models, the concept of duration models is used as the alignment module of two sequences to replace attention, which can avoid the uncertainty in attention, and significantly improve the stability of the seq2saq models.

#### FastSpeech
Instead of using the encoder-attention-decoder based architecture as adopted by most seq2seq based autoregressive and non-autoregressive generation, [FastSpeech](https://arxiv.org/abs/1905.09263) is a novel feed-forward structure, which can generate a target mel spectrogram sequence in parallel.

**Features of FastSpeech:**
- Encoder: based on Transformer.
- Change `FFN` to `CNN` in self-attention.
    -  Model local dependency.
- Length regulator.
    - Use real phoneme durations to expand the output frame of the encoder during training.
- Non-autoregressive decode.
    -  Improve generation efficiency.

**Length predictor:**
- Pretrain a TransformerTTS model.
- Get alignment matrix of train data.
- Calculate the phoneme durations according to the probability of the alignment matrix.
- Use the output of the encoder to predict the phoneme durations and calculate the MSE loss.
- Use real phoneme durations to expand the output frame of the encoder during training.
- Use phoneme durations predicted by the duration model to expand the frame during prediction.
    - Attentrion can not control phoneme durations. The explicit duration modeling can control durations through duration coefficient (duration coefficient is `1` during training).

**Advantages of non-autoregressive decoder:**
- The built-in duration model of the seq2seq model has converted the input length `M` to the output length `N`.
- The length of the output is known, `stop token` is no longer used, avoiding the problem of being unable to stop.
• Can be generated in parallel (decoding time is less affected by sequence length)

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/fastspeech.png" width=800 /> <br>
</div>

#### FastPitch
[FastPitch](https://arxiv.org/abs/2006.06873) follows FastSpeech. A single pitch value is predicted for every temporal location, which improves the overall quality of synthesized speech.

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/fastpitch.png" width=500 /> <br>
</div>

#### FastSpeech2
**Disadvantages of FastSpeech:**
- The teacher-student distillation pipeline is complicated and time-consuming.
- The duration extracted from the teacher model is not accurate enough.
- The target mel spectrograms distilled from the teacher model suffer from information loss due to data simplification.

[FastSpeech2](https://arxiv.org/abs/2006.04558)  addresses the issues in FastSpeech and better solves the one-to-many mapping problem in TTS.

**Features of FastSpeech2:**
- Directly train the model with the ground-truth target instead of the simplified output from the teacher.
- Introducing more variation information of speech as conditional inputs, extract `duration`, `pitch`, and `energy` from speech waveform and directly take them as conditional inputs in training and use predicted values in inference.

FastSpeech2 is similar to FastPitch but introduces more variation information of the speech.

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/fastspeech2.png" width=800 /> <br>
</div>

You can find PaddleSpeech TTS's FastSpeech2/FastPitch with CSMSC dataset example at [examples/csmsc/tts3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/tts3), We use token-averaged pitch and energy values introduced in FastPitch rather than frame-level ones in FastSpeech2.

### SpeedySpeech
[SpeedySpeech](https://arxiv.org/abs/2008.03802) simplify the teacher-student architecture of FastSpeech and provide a fast and stable training procedure.

**Features of SpeedySpeech:**
- Use a simpler, smaller, and faster-to-train convolutional teacher model ([Deepvoice3](https://arxiv.org/abs/1710.07654) and [DCTTS](https://arxiv.org/abs/1710.08969)) with a single attention layer instead of Transformer used in FastSpeech.  
- Show that self-attention layers in the student network are not needed for high-quality speech synthesis.
- Describe a simple data augmentation technique that can be used early in the training to make the teacher network robust to sequential error propagation.

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/speedyspeech.png" width=500 /> <br>
</div>

You can find PaddleSpeech TTS's SpeedySpeech with CSMSC dataset example at [examples/csmsc/tts2](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/tts2).

## Vocoders
In speech synthesis, the main task of the vocoder is to convert the spectral parameters predicted by the acoustic model into the final speech waveform.

Taking into account the short-term change frequency of the waveform, the acoustic model usually avoids direct modeling of the speech waveform, but firstly models the spectral features extracted from the speech waveform, and then reconstructs the waveform by the decoding part of the vocoder.

A vocoder usually consists of a pair of encoders and decoders for speech analysis and synthesis. The encoder estimates the parameters, and then the decoder restores the speech.

Vocoders based on neural networks usually is speech synthesis, which learns the mapping relationship from spectral features to waveforms through training data.

### Categories of neural vocodes
- Autoregression
    - WaveNet
    - WaveRNN
    - LPCNet

- Flow
    - **WaveFlow**
    - WaveGlow
    - FloWaveNet
    - Parallel WaveNet
- GAN
    - WaveGAN
    - **Parallel WaveGAN**
    - **MelGAN**
    - **Style MelGAN**
    - **Multi Band MelGAN**
    - **HiFi GAN**
- VAE
    - Wave-VAE
- Diffusion
    - WaveGrad
    - DiffWave

**Motivations of GAN-based vocoders:**
- Modeling speech signals by estimating probability distribution usually has high requirements for the expression ability of the model itself. In addition, specific assumptions need to be made about the distribution of waveforms.
- Although autoregressive neural vocoders can obtain high-quality synthetic speech, such models usually have a **slow generation speed**.
- The training of inverse autoregressive flow vocoders is complex, and they also require the modeling capability of long-term context information.
- Vocoders based on Bipartite Transformation converge slowly and are complex.
- GAN-based vocoders don't need to make assumptions about the speech distribution and train through adversarial learning.

Here, we introduce a Flow-based vocoder WaveFlow and a GAN-based vocoder Parallel WaveGAN.

### WaveFlow
 [WaveFlow](https://arxiv.org/abs/1912.01219) is proposed by Baidu Research.

**Features of WaveFlow:**
- It can synthesize 22.05 kHz high-fidelity speech around 40x faster than real-time on an Nvidia V100 GPU without engineered inference kernels, which is faster than [WaveGlow](https://github.com/NVIDIA/waveglow) and several orders of magnitude faster than WaveNet.
- It is a small-footprint flow-based model for raw audio. It has only 5.9M parameters, which is 15x smaller than WaveGlow (87.9M).
- It is directly trained with maximum likelihood without probability density distillation and auxiliary losses as used in [Parallel WaveNet](https://arxiv.org/abs/1711.10433) and [ClariNet](https://openreview.net/pdf?id=HklY120cYm), which simplifies the training pipeline and reduces the cost of development.

You can find PaddleSpeech TTS's WaveFlow with LJSpeech dataset example at [examples/ljspeech/voc0](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/voc0).

### Parallel WaveGAN
[Parallel WaveGAN](https://arxiv.org/abs/1910.11480) trains a non-autoregressive WaveNet variant as a generator in a GAN-based training method.

**Features of Parallel WaveGAN:**

- Use non-causal convolution instead of causal convolution.
- The input is random Gaussian white noise.
- The model is non-autoregressive both in training and prediction, which is fast
- Multi-resolution STFT loss.

<div align="left">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSpeech/develop/docs/images/pwg.png" width=600 /> <br>
</div>

You can find PaddleSpeech TTS's Parallel WaveGAN with CSMSC example at [examples/csmsc/voc1](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc1).
