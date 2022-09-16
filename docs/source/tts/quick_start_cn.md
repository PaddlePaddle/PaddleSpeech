(简体中文|[English](./quick_start.md))
# 语音合成快速开始
这些PaddleSpeech中的样例主要按数据集分类，我们主要使用的TTS数据集有：

* CSMCS (普通话单发音人)
* AISHELL3 (普通话多发音人)
* LJSpeech (英文单发音人)
* VCTK (英文多发音人)

PaddleSpeech 的 TTS 模型具有以下映射关系：

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

## 快速开始

让我们以 FastSpeech2 + Parallel WaveGAN 和 CSMSC 数据集 为例. [examples/csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc)

### 用 CSMSC 数据集训练 Parallel WaveGAN

- 进入目录
    ```bash
    cd examples/csmsc/voc1
    ```
- 设置环境变量
    ```bash
    source path.sh
    ```
    **在你开始做任何事情之前，必须先做这步**
    将 `MAIN_ROOT` 设置为项目目录. 使用 `parallelwave_gan` 模型作为 `MODEL`.

- 运行
    ```bash
    bash run.sh
    ```
    这只是一个演示，请确保源数据已经准备好，并且在下一个 `step` 之前每个 `step` 都运行正常.
### 用CSMSC数据集训练FastSpeech2

- 进入目录
    ```bash
    cd examples/csmsc/tts3
    ```
    
- 设置环境变量
    ```bash
    source path.sh
    ```
    **在你开始做任何事情之前，必须先做这步**
    将 `MAIN_ROOT` 设置为项目目录. 使用 `fastspeech2` 模型作为 `MODEL` 。
    
- 运行
    ```bash
    bash run.sh
    ```
    这只是一个演示，请确保源数据已经准备好，并且在下一个 `step` 之前每个 `step` 都运行正常。

`run.sh` 中主要包括以下步骤：

- 设置路径。
- 预处理数据集，
- 训练模型。
- 从 `metadata.jsonl` 中合成波形
- 从文本文件合成波形。（在声学模型中）
- 使用静态模型进行推理。（可选）

有关更多详细信息，请参见 examples 中的 `README.md`

## TTS 流水线
本节介绍如何使用 TTS 提供的预训练模型，并对其进行推理。

TTS中的预训练模型在压缩包中提供。将其解压缩以获得如下文件夹：
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
- `default.yaml` 存储用于训练模型的配置。
- `snapshot_iter_*.pdz` 是检查点文件，其中`*`是它经过训练的步骤。
- `*_stats.npy` 是特征的统计文件，如果它在训练前已被标准化。
- `phone_id_map.txt` 是音素到音素 ID 的映射关系。
- `tone_id_map.txt` 是在训练声学模型之前分割音调和拼音时，音调到音调 ID 的映射关系。（例如在 csmsc/speedyspeech 的示例中）
- `spk_id_map.txt` 是多发音人声学模型中 "发音人" 到 "spk_ids" 的映射关系。

下面的示例代码显示了如何使用模型进行预测。
### Acoustic Models 声学模型（文本到频谱图）
下面的代码显示了如何使用 `FastSpeech2` 模型。加载预训练模型后，使用它和 normalizer 对象构建预测对象，然后使用 `fastspeech2_inferencet(phone_ids)` 生成频谱图，频谱图可进一步用于使用声码器合成原始音频。

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

# 加载预训练模型
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

# 加载特征文件
stat = np.load(checkpoint_dir / "speech_stats.npy")
mu, std = stat
mu = paddle.to_tensor(mu)
std = paddle.to_tensor(std)
fastspeech2_normalizer = ZScore(mu, std)

# 构建预测对象
fastspeech2_inference = FastSpeech2Inference(fastspeech2_normalizer, model)

# load Chinese Frontend
frontend = Frontend(checkpoint_dir / "phone_id_map.txt")

# 构建一个中文前端
sentence = "你好吗？"
input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
phone_ids = input_ids["phone_ids"]
flags = 0
# 构建预测对象加载中文前端，对中文文本前端的输出进行分段
for part_phone_ids in phone_ids:
    with paddle.no_grad():
        temp_mel = fastspeech2_inference(part_phone_ids)
        if flags == 0:
            mel = temp_mel
            flags = 1
        else:
            mel = paddle.concat([mel, temp_mel])
```

### Vcoder声码器（谱图到波形）
下面的代码显示了如何使用 `Parallel WaveGAN` 模型。像上面的例子一样，加载预训练模型后，使用它和 normalizer 对象构建预测对象，然后使用 `pwg_inference(mel)` 生成原始音频（ wav 格式）。

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

# 加载预训练模型
checkpoint_dir = Path("parallel_wavegan_baker_ckpt_0.4")
with open(checkpoint_dir / "pwg_default.yaml") as f:
    pwg_config = CfgNode(yaml.safe_load(f))
vocoder = PWGGenerator(**pwg_config["generator_params"])
vocoder.set_state_dict(paddle.load(args.pwg_params))
vocoder.remove_weight_norm()
vocoder.eval()

# 加载特征文件
stat = np.load(checkpoint_dir / "pwg_stats.npy")
mu, std = stat
mu = paddle.to_tensor(mu)
std = paddle.to_tensor(std)
pwg_normalizer = ZScore(mu, std)

# 加载预训练模型构造预测对象
pwg_inference = PWGInference(pwg_normalizer, vocoder)

# 频谱图到波形
wav = pwg_inference(mel)
sf.write(
        audio_path,
        wav.numpy(),
        samplerate=fastspeech2_config.fs)
```
