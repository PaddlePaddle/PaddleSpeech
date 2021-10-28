# Audio Tagging

声音分类的任务是单标签的分类任务，但是对于一段音频来说，它可以是多标签的。譬如在一般的室内办公环境进行录音，这段音频里可能包含人们说话的声音、键盘敲打的声音、鼠标点击的声音，还有室内的一些其他背景声音。对于通用的声音识别和声音检测场景而言，对一段音频预测多个标签是具有很强的实用性的。

在IEEE ICASSP 2017 大会上，谷歌开放了一个大规模的音频数据集[Audioset](https://research.google.com/audioset/)。该数据集包含了 632 类的音频类别以及 2,084,320 条人工标记的每段 10 秒长度的声音剪辑片段（来源于YouTube视频）。目前该数据集已经有210万个已标注的视频数据，5800小时的音频数据，经过标记的声音样本的标签类别为527。

`PANNs`([PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211.pdf))是基于Audioset数据集训练的声音分类/识别的模型。其预训练的任务是多标签的声音识别，因此可用于声音的实时tagging。

本示例采用`PANNs`预训练模型，基于Audioset的标签类别对输入音频实时tagging，并最终以文本形式输出对应时刻的top k类别和对应的得分。


## 模型简介

PaddleAudio提供了PANNs的CNN14、CNN10和CNN6的预训练模型，可供用户选择使用：
- CNN14: 该模型主要包含12个卷积层和2个全连接层，模型参数的数量为79.6M，embbedding维度是2048。
- CNN10: 该模型主要包含8个卷积层和2个全连接层，模型参数的数量为4.9M，embbedding维度是512。
- CNN6: 该模型主要包含4个卷积层和2个全连接层，模型参数的数量为4.5M，embbedding维度是512。


## 快速开始

### 模型预测

```shell
export CUDA_VISIBLE_DEVICES=0
python audio_tag.py --device gpu --wav ./cat.wav --sample_duration 2 --hop_duration 0.3 --output_dir ./output_dir
```

可支持配置的参数：

- `device`: 选用什么设备进行训练，可选cpu或gpu，默认为gpu。如使用gpu训练则参数gpus指定GPU卡号。
- `wav`: 指定预测的音频文件。
- `sample_duration`: 模型每次预测的音频时间长度，单位为秒，默认为2s。
- `hop_duration`: 每两个预测音频的时间间隔，单位为秒，默认为0.3s。
- `output_dir`: 模型预测结果存放的路径，默认为`./output_dir`。

示例代码中使用的预训练模型为`CNN14`，如果想更换为其他预训练模型，可通过以下方式执行：
```python
from paddleaudio.models.panns import cnn14, cnn10, cnn6

# CNN14
model = cnn14(pretrained=True, extract_embedding=False)
# CNN10
model = cnn10(pretrained=True, extract_embedding=False)
# CNN6
model = cnn6(pretrained=True, extract_embedding=False)
```

执行结果：
```
[2021-04-30 19:15:41,025] [    INFO] - Saved tagging results to ./output_dir/audioset_tagging_sr_44100.npz
```

执行后得分结果保存在`output_dir`的`.npz`文件中。


### 生成tagging标签文本
```shell
python parse_result.py --tagging_file ./output_dir/audioset_tagging_sr_44100.npz --top_k 10 --smooth True --smooth_size 5 --label_file ./assets/audioset_labels.txt --output_dir ./output_dir
```

可支持配置的参数：

- `tagging_file`: 模型预测结果文件。
- `top_k`: 获取预测结果中，得分最高的前top_k个标签，默认为10。
- `smooth`: 预测结果的后验概率平滑，默认为True，表示应用平滑。
- `smooth_size`: 平滑计算过程中的样本数量，默认为5。
- `label_file`: 模型预测结果对应的Audioset类别的文本文件。
- `output_dir`: 标签文本存放的路径，默认为`./output_dir`。

执行结果：
```
[2021-04-30 19:26:58,743] [    INFO] - Posterior smoothing...
[2021-04-30 19:26:58,746] [    INFO] - Saved tagging labels to ./output_dir/audioset_tagging_sr_44100.txt
```

执行后文本结果保存在`output_dir`的`.txt`文件中。


## Tagging标签文本

最终输出的文本结果如下所示。  
样本每个时间范围的top k结果用空行分隔。在每一个结果中，第一行是时间信息，数字表示tagging结果在时间起点信息，比例值代表当前时刻`t`与音频总长度`T`的比值；紧接的k行是对应的标签和得分。

```
0.0
Cat: 0.9144676923751831
Animal: 0.8855036497116089
Domestic animals, pets: 0.804577112197876
Meow: 0.7422927021980286
Music: 0.19959309697151184
Inside, small room: 0.12550437450408936
Caterwaul: 0.021584441885352135
Purr: 0.020247288048267365
Speech: 0.018197158351540565
Vehicle: 0.007446660194545984

0.059197544398158296
Cat: 0.9250872135162354
Animal: 0.8957151174545288
Domestic animals, pets: 0.8228275775909424
Meow: 0.7650775909423828
Music: 0.20210561156272888
Inside, small room: 0.12290887534618378
Caterwaul: 0.029371455311775208
Purr: 0.018731823191046715
Speech: 0.017130598425865173
Vehicle: 0.007748497650027275

0.11839508879631659
Cat: 0.9336574673652649
Animal: 0.9111202359199524
Domestic animals, pets: 0.8349071145057678
Meow: 0.7761964797973633
Music: 0.20467285811901093
Inside, small room: 0.10709915310144424
Caterwaul: 0.05370649695396423
Purr: 0.018830426037311554
Speech: 0.017361722886562347
Vehicle: 0.006929398979991674

...
...
```

以下[Demo](https://bj.bcebos.com/paddleaudio/media/audio_tagging_demo.mp4)展示了一个将tagging标签输出到视频的例子，可以实时地对音频进行多标签预测。

![](https://bj.bcebos.com/paddleaudio/media/audio_tagging_demo.gif)
