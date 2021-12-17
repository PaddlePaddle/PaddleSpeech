# Style FastSpeech2
## Introduction
[FastSpeech2](https://arxiv.org/abs/2006.04558)  is a classical acoustic model for Text-to-Speech synthesis, which introduces controllable speech input, including `phoneme duration`、 `energy` and `pitch`. 

In the prediction phase, you can change these controllable variables to get some interesting results.

For example:

1. The `duration` control in `FastSpeech2` can control the speed of audios will keep the `pitch`. (in some speech tools, increasing the speed will increase the pitch and vice versa.)

2. When we set the `pitch` of one sentence to a mean value and set the `tones` of phones to `1`, we will get a `robot-style` timbre.

3. When we raise the `pitch` of an adult female (with a fixed scale ratio), we will get a `child-style` timbre.

The `duration` and `pitch` of different phonemes in a sentence can have different scale ratios. You can set different scale ratios to emphasize or weaken the pronunciation of some phonemes.
## Usage
Run the following command line to get started:
```
./run.sh
```
In `run.sh`, it will execute `source path.sh` firstly, which will set the environment variants.

If you would like to try your sentence, please replace the sentence in `sentences.txt`.

For more details, please see `style_syn.py`

The audio samples are in [style-control-in-fastspeech2](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html#style-control-in-fastspeech2)
