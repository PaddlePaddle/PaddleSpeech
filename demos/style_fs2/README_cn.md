(简体中文|[English](./README.md))

# Style FastSpeech2

## 简介

[FastSpeech2](https://arxiv.org/abs/2006.04558)  是用于语音合成的经典声学模型，它引入了可控语音输入，包括 `phoneme duration` 、 `energy` 和 `pitch` 。

在预测阶段，您可以更改这些变量以获得一些有趣的结果。

例如:

1.  `FastSpeech2` 中的 `duration` 可以控制音频的速度 ，并保持 `pitch` 。（在某些语音工具中，增加速度将增加音调，反之亦然。）
2. 当我们将一个句子的 `pitch` 设置为平均值并将音素的 `tones` 设置为 `1` 时，我们将获得 `robot-style` 的音色。
3. 当我们提高成年女性的 `pitch` （比例固定）时，我们会得到 `child-style` 的音色。

句子中不同音素的 `duration` 和 `pitch` 可以具有不同的比例。您可以设置不同的音阶比例来强调或削弱某些音素的发音。

## 运行

运行以下命令行开始：

```
./run.sh
```

在 `run.sh`, 会首先执行 `source path.sh` 去设置好环境变量。

如果您想尝试您的句子，请替换 `sentences.txt`中的句子。

更多的细节，请查看 `style_syn.py`。

语音样例可以在 [style-control-in-fastspeech2](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html#style-control-in-fastspeech2) 查看。
