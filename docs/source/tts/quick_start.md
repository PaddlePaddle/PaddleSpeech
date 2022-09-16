([简体中文](./quick_start_cn.md)|English)
# Quick Start of Text-to-Speech
The examples in PaddleSpeech are mainly classified by datasets, the TTS datasets we mainly used are:
* CSMCS (Mandarin single speaker)
* AISHELL3 (Mandarin multiple speakers)
* LJSpeech (English single speaker)
* VCTK (English multiple speakers)

The models in PaddleSpeech TTS have the following mapping relationship:
* tts0 - Tactron2
* tts1 - TransformerTTS
* tts2 - SpeedySpeech
* tts3 - FastSpeech2
* voc0 - WaveFlow
* voc1 - Parallel WaveGAN
* voc2 - MelGAN
* voc3 - MultiBand MelGAN
* voc4 - Style MelGAN
* voc5 - HiFiGAN
* vc0 - Tactron2 Voice Clone with GE2E
* vc1 - FastSpeech2 Voice Clone with GE2E

## Quick Start

Let's take a FastSpeech2 + Parallel WaveGAN with CSMSC dataset for instance. [examples/csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc)

### Train Parallel WaveGAN with CSMSC
- Go to the directory
    ```bash
    cd examples/csmsc/voc1
    ```
- Source env
    ```bash
    source path.sh
    ```
    **Must do this before you start to do anything.**
    Set `MAIN_ROOT` as project dir. Using `parallelwave_gan` model as `MODEL`.

- Main entrypoint
    ```bash
    bash run.sh
    ```
    This is just a demo, please make sure source data have been prepared well and every `step` works well before the next `step`.
### Train FastSpeech2 with CSMSC
- Go to the directory
    ```bash
    cd examples/csmsc/tts3
    ```
- Source env
    ```bash
    source path.sh
    ```
    **Must do this before you start to do anything.**
    Set `MAIN_ROOT` as project dir. Using `fastspeech2` model as `MODEL`.
- Main entry point
    ```bash
    bash run.sh
    ```
    This is just a demo, please make sure source data have been prepared well and every `step` works well before the next `step`.

The steps in `run.sh` mainly include:
- source path.
- preprocess the dataset,
- train the model.
- synthesize waveform from metadata.jsonl.
- synthesize waveform from a text file. (in acoustic models)
- inference using a static model. (optional)

For more details, you can see `README.md` in examples.

## Pipeline of TTS
This section shows how to use pretrained models provided by TTS and make an inference with them.

Pretrained models in TTS are provided in an archive. Extract it to get a folder like this:
**Acoustic Models:**
```text
checkpoint_name
├── default.yaml
├── snapshot_iter_*.pdz
├── speech_stats.npy
├── phone_id_map.txt
├── spk_id_map.txt (optimal)
└── tone_id_map.txt (optimal)
```
**Vocoders:**
```text
checkpoint_name
├── default.yaml  
├── snapshot_iter_*.pdz
└── stats.npy  
```
- `default.yaml` stores the config used to train the model.
- `snapshot_iter_*.pdz` is the checkpoint file, where `*` is the steps it has been trained.
- `*_stats.npy` is the stats file of the feature if it has been normalized before training.
- `phone_id_map.txt` is the map of phonemes to phoneme_ids.
- `tone_id_map.txt` is the map of tones to tones_ids, when you split tones and phones before training acoustic models. (for example in our csmsc/speedyspeech example)
- `spk_id_map.txt` is the map of speakers to spk_ids in multi-spk acoustic models. (for example in our aishell3/fastspeech2 example)

The example code below shows how to use the models for prediction.
### Acoustic Models (text to spectrogram)
The code below shows how to use a `FastSpeech2` model.  After loading the pretrained model, use it and the normalizer object to construct a prediction object，then use `fastspeech2_inferencet(phone_ids)` to generate spectrograms, which can be further used to synthesize raw audio with a vocoder.

```python
from pathlib import Path
import numpy as np
import paddle
import yaml
from yacs.config import CfgNode
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Inference
from paddlespeech.t2s.modules.normalizer import ZScore
# examples/fastspeech2/baker/frontend.py
from frontend import Frontend

# load the pretrained model
checkpoint_dir = Path("fastspeech2_nosil_baker_ckpt_0.4")
with open(checkpoint_dir / "phone_id_map.txt", "r") as f:
    phn_id = [line.strip().split() for line in f.readlines()]
vocab_size = len(phn_id)
with open(checkpoint_dir / "default.yaml") as f:
    fastspeech2_config = CfgNode(yaml.safe_load(f))
odim = fastspeech2_config.n_mels
model = FastSpeech2(
    idim=vocab_size, odim=odim, **fastspeech2_config["model"])
model.set_state_dict(
    paddle.load(args.fastspeech2_checkpoint)["main_params"])
model.eval()

# load stats file
stat = np.load(checkpoint_dir / "speech_stats.npy")
mu, std = stat
mu = paddle.to_tensor(mu)
std = paddle.to_tensor(std)
fastspeech2_normalizer = ZScore(mu, std)

# construct a prediction object
fastspeech2_inference = FastSpeech2Inference(fastspeech2_normalizer, model)

# load Chinese Frontend
frontend = Frontend(checkpoint_dir / "phone_id_map.txt")

# text to spectrogram
sentence = "你好吗？"
input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
phone_ids = input_ids["phone_ids"]
flags = 0
# The output of Chinese text frontend is segmented
for part_phone_ids in phone_ids:
    with paddle.no_grad():
        temp_mel = fastspeech2_inference(part_phone_ids)
        if flags == 0:
            mel = temp_mel
            flags = 1
        else:
            mel = paddle.concat([mel, temp_mel])
```

### Vocoder (spectrogram to wave)
The code below shows how to use a  ` Parallel WaveGAN` model. Like the example above, after loading the pretrained model, use it and the normalizer object to construct a prediction object，then use `pwg_inference(mel)` to generate raw audio (in wav format).

```python
from pathlib import Path
import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.parallel_wavegan import PWGInference
from paddlespeech.t2s.modules.normalizer import ZScore

# load the pretrained model
checkpoint_dir = Path("parallel_wavegan_baker_ckpt_0.4")
with open(checkpoint_dir / "pwg_default.yaml") as f:
    pwg_config = CfgNode(yaml.safe_load(f))
vocoder = PWGGenerator(**pwg_config["generator_params"])
vocoder.set_state_dict(paddle.load(args.pwg_params))
vocoder.remove_weight_norm()
vocoder.eval()

# load stats file
stat = np.load(checkpoint_dir / "pwg_stats.npy")
mu, std = stat
mu = paddle.to_tensor(mu)
std = paddle.to_tensor(std)
pwg_normalizer = ZScore(mu, std)

# construct a prediction object
pwg_inference = PWGInference(pwg_normalizer, vocoder)

# spectrogram to wave
wav = pwg_inference(mel)
sf.write(
        audio_path,
        wav.numpy(),
        samplerate=fastspeech2_config.fs)
```
