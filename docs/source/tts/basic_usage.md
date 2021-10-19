# Basic Usage
This section shows how to use pretrained models provided by parakeet and make inference with them.

Pretrained models in v0.4 are provided in a archive. Extract it to get a folder like this:
```
checkpoint_name/
├──default.yaml
├──snapshot_iter_76000.pdz
├──speech_stats.npy
└──phone_id_map.txt
```
`default.yaml` stores the config used to train the model.
`snapshot_iter_N.pdz` is the chechpoint file, where `N` is the steps it has been trained.
`*_stats.npy` is the stats file of feature if  it has been normalized before training.
`phone_id_map.txt` is the map of  phonemes to phoneme_ids.

The example code below shows how to use the models for prediction.
## Acoustic Models (text to spectrogram)
The code below show how to use a `FastSpeech2` model.  After loading the pretrained model, use it and normalizer object to construct a prediction object，then use fastspeech2_inferencet(phone_ids) to generate spectrograms, which can be further used to synthesize raw audio with a vocoder.

```python
from pathlib import Path
import numpy as np
import paddle
import yaml
from yacs.config import CfgNode
from parakeet.models.fastspeech2 import FastSpeech2
from parakeet.models.fastspeech2 import FastSpeech2Inference
from parakeet.modules.normalizer import ZScore
# Parakeet/examples/fastspeech2/baker/frontend.py
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

## Vocoder (spectrogram to wave)
The code below show how to use a  ` Parallel WaveGAN` model. Like the example above, after loading the pretrained model, use it and normalizer object to construct a prediction object，then use pwg_inference(mel) to generate  raw audio (in wav format).

```python
from pathlib import Path
import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode
from parakeet.models.parallel_wavegan import PWGGenerator
from parakeet.models.parallel_wavegan import PWGInference
from parakeet.modules.normalizer import ZScore

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
