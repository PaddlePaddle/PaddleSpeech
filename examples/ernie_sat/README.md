ERNIE-SAT 是可以同时处理中英文的跨语言的语音-语言跨模态大模型，其在语音编辑、个性化语音合成以及跨语言的语音合成等多个任务取得了领先效果。可以应用于语音编辑、个性化合成、语音克隆、同传翻译等一系列场景，该项目供研究使用。

## 模型框架
ERNIE-SAT 中我们提出了两项创新：
- 在预训练过程中将中英双语对应的音素作为输入，实现了跨语言、个性化的软音素映射
- 采用语言和语音的联合掩码学习实现了语言和语音的对齐

[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-3lOXKJXE-1655380879339)(.meta/framework.png)]

## 使用说明

### 1.安装飞桨与环境依赖

- 本项目的代码基于 Paddle(version>=2.0)
- 本项目开放提供加载 torch 版本的 vocoder 的功能
  - torch version>=1.8

- 安装 htk: 在[官方地址](https://htk.eng.cam.ac.uk/)注册完成后，即可进行下载较新版本的 htk (例如 3.4.1)。同时提供[历史版本 htk 下载地址](https://htk.eng.cam.ac.uk/ftp/software/)

    - 1.注册账号，下载 htk
    - 2.解压 htk 文件，**放入项目根目录的 tools 文件夹中, 以 htk 文件夹名称放入**
    - 3.**注意**: 如果您下载的是 3.4.1 或者更高版本, 需要进入 HTKLib/HRec.c 文件中, **修改 1626 行和 1650 行**, 即把**以下两行的 dur<=0 都修改为 dur<0**，如下所示:
        ```bash
         以htk3.4.1版本举例: 
         (1)第1626行: if (dur<=0 && labid != splabid) HError(8522,"LatFromPaths: Align  have dur<=0");
         修改为:      if (dur<0 && labid != splabid) HError(8522,"LatFromPaths: Align  have dur<0");

         (2)1650行: if (dur<=0 && labid != splabid) HError(8522,"LatFromPaths: Align have dur<=0 ");
         修改为:     if (dur<0 && labid != splabid) HError(8522,"LatFromPaths: Align have dur<0 ");
        ```
    - 4.**编译**: 详情参见解压后的 htk 中的 README 文件(如果未编译, 则无法正常运行)
     


- 安装 ParallelWaveGAN: 参见[官方地址](https://github.com/kan-bayashi/ParallelWaveGAN)：按照该官方链接的安装流程，直接在**项目的根目录下** git clone ParallelWaveGAN 项目并且安装相关依赖即可。


- 安装其他依赖: **sox, libsndfile**等

### 2.预训练模型
预训练模型 ERNIE-SAT 的模型如下所示:
- [ERNIE-SAT_ZH](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/old/model-ernie-sat-base-zh.tar.gz) 
- [ERNIE-SAT_EN](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/old/model-ernie-sat-base-en.tar.gz)  
- [ERNIE-SAT_ZH_and_EN](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/old/model-ernie-sat-base-en_zh.tar.gz) 


创建 pretrained_model 文件夹，下载上述 ERNIE-SAT 预训练模型并将其解压: 
```bash
mkdir pretrained_model
cd pretrained_model
tar -zxvf model-ernie-sat-base-en.tar.gz
tar -zxvf model-ernie-sat-base-zh.tar.gz
tar -zxvf model-ernie-sat-base-en_zh.tar.gz
```

### 3.下载

1. 本项目使用 parallel wavegan 作为声码器（vocoder）: 
    - [pwg_aishell3_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip)  

    创建 download 文件夹，下载上述预训练的声码器（vocoder）模型并将其解压:

    ```bash
    mkdir download
    cd download
    unzip pwg_aishell3_ckpt_0.5.zip
    ```

2. 本项目使用 [FastSpeech2](https://arxiv.org/abs/2006.04558) 作为音素（phoneme）的持续时间预测器:
    - [fastspeech2_conformer_baker_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_conformer_baker_ckpt_0.5.zip)  中文场景下使用 
    - [fastspeech2_nosil_ljspeech_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_ljspeech_ckpt_0.5.zip)  英文场景下使用

    下载上述预训练的 fastspeech2 模型并将其解压:

    ```bash
    cd download
    unzip fastspeech2_conformer_baker_ckpt_0.5.zip
    unzip fastspeech2_nosil_ljspeech_ckpt_0.5.zip
    ```

3. 本项目使用 HTK 获取输入音频和文本的对齐信息:
	
	- [aligner.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/old/aligner.zip) 

	下载上述文件到 tools 文件夹并将其解压:
    ```bash
    cd tools
    unzip aligner.zip
    ```


### 4.推理

本项目当前开源了语音编辑、个性化语音合成、跨语言语音合成的推理代码，后续会逐步开源。
注：当前英文场下的合成语音采用的声码器默认为 vctk_parallel_wavegan.v1.long, 可在[该链接](https://github.com/kan-bayashi/ParallelWaveGAN)中找到; 若 use_pt_vocoder 参数设置为 False，则英文场景下使用 paddle 版本的声码器。

我们提供特定音频文件, 以及其对应的文本、音素相关文件:
- prompt_wav: 提供的音频文件
- prompt/dev: 基于上述特定音频对应的文本、音素相关文件


```text
prompt_wav
├── p299_096.wav                 # 样例语音文件1
├── p243_313.wav                 # 样例语音文件2
└── ...
```

```text
prompt/dev
├── text                     # 样例语音对应文本
├── wav.scp                  # 样例语音路径
├── mfa_text                 # 样例语音对应音素
├── mfa_start                # 样例语音中各个音素的开始时间
└── mfa_end                  # 样例语音中各个音素的结束时间
```
1. `--am` 声学模型格式符合 {model_name}_{dataset}
2. `--am_config`, `--am_checkpoint`, `--am_stat` 和 `--phones_dict` 是声学模型的参数，对应于 fastspeech2 预训练模型中的 4 个文件。
3. `--voc` 声码器(vocoder)格式是否符合 {model_name}_{dataset}
4. `--voc_config`, `--voc_checkpoint`, `--voc_stat` 是声码器的参数，对应于 parallel wavegan 预训练模型中的 3 个文件。
5. `--lang` 对应模型的语言可以是 `zh` 或 `en` 。
6. `--ngpu` 要使用的 GPU 数，如果 ngpu==0，则使用 cpu。
7. `--model_name` 模型名称
8. `--uid` 特定提示(prompt)语音的 id
9. `--new_str` 输入的文本(本次开源暂时先设置特定的文本)
10. `--prefix` 特定音频对应的文本、音素相关文件的地址
11. `--source_lang` , 源语言
12. `--target_lang` , 目标语言
13. `--output_name` , 合成语音名称
14. `--task_name` , 任务名称, 包括：语音编辑任务、个性化语音合成任务、跨语言语音合成任务

运行以下脚本即可进行实验
```shell
./run_sedit_en.sh       # 语音编辑任务(英文) 
./run_gen_en.sh         # 个性化语音合成任务(英文)
./run_clone_en_to_zh.sh # 跨语言语音合成任务(英文到中文的语音克隆)
```
