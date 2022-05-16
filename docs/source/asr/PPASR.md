([简体中文](./PPASR_cn.md)|English)
# PP-ASR

## Catalogue
- [1. Introduction](#1)
- [2. Characteristic](#2)
- [3. Tutorials](#3)
    - [3.1 Pre-trained Models](#31)
    - [3.2 Training](#32)
    - [3.3 Inference](#33)
    - [3.4 Service Deployment](#33)
    - [3.5 Customized Auto Speech Recognition and Deployment](#33)
- [4. Quick Start](#4)

<a name="1"></a>
## 1. Introduction

PP-ASR is a tool to provide ASR(Automatic speech recognition) function. It provides a variety of Chinese and English models and supports model training. It also supports model inference using the command line. In addition, PP-ASR supports the deployment of streaming models and customized ASR.

<a name="2"></a>
## 2. Characteristic
The basic process of ASR is shown in the figure below:  
<center><img src=https://user-images.githubusercontent.com/87408988/168259962-cbe2008b-47b6-443d-9566-d77a5ca2eb25.png width="800" ></center>


The main characteristics of PP-ASR are shown below:
-  Provides pre-trained models on Chinese/English open source datasets: aishell(Chinese), wenetspeech(Chinese) and librispeech(English). The models include deepspeech2 and conformer/transformer.
-  Support model training on Chinese/English datasets.
-  Support model inference using the command line. You can use to use `paddlespeech asr --model xxx --input xxx.wav` to use the pre-trained model to do model inference. 
-  Support deployment of streaming ASR server. Besides ASR function, the server supports timestamp function.
-  Support customized auto speech recognition and deployment.

<a name="3"></a>
## 3. Tutorials

<a name="31"></a>
## 3.1 Pre-trained Models
The support pre-trained model list: [released_model](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md).  
The model with good effect are Ds2 Online Wenetspeech ASR0 Model and Conformer Online Wenetspeech ASR1 Model. Both two models support streaming ASR.  
For more information about model design, you can refer to the aistudio tutorial:
- [Deepspeech2](https://aistudio.baidu.com/aistudio/projectdetail/3866807)
- [Transformer](https://aistudio.baidu.com/aistudio/projectdetail/3470110)

<a name="32"></a>
## 3.2 Training
The referenced script for model training is stored in [examples](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples) and stored according to "examples/dataset/model". The dataset mainly supports aishell and librispeech. The model supports deepspeech2 and u2(conformer/transformer).
The specific steps of executing the script are recorded in `run.sh`.

For more information, you can refer to [asr1](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell/asr1)


<a name="33"></a>
## 3.3 Inference

PP-ASR supports use `paddlespeech asr --model xxx --input xxx.wav` to use the pre-trained model to do model inference after install `paddlespeech` by `pip install paddlespeech`.

Specific supported functions include:

- Prediction of single audio
- Use the pipe to predict multiple audio
- Support RTF calculation

For specific usage, please refer to: [speech_recognition](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/speech_recognition/README_cn.md) 


<a name="34"></a>
## 3.4 Service Deployment

PP-ASR supports the service deployment of streaming ASR. Support the simultaneous use of speech recognition and punctuation processing.

Demo of ASR Server: [streaming_asr_server](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/streaming_asr_server)

![image](https://user-images.githubusercontent.com/87408988/168255342-1fc790c0-16f4-4540-a861-db239076727c.png)

Display of using ASR server on Web page: [streaming_asr_demo_video](https://paddlespeech.readthedocs.io/en/latest/streaming_asr_demo_video.html)


For more information about service deployment, you can refer to the aistudio tutorial:
- [Streaming service - model part](https://aistudio.baidu.com/aistudio/projectdetail/3839884)
- [Streaming service](https://aistudio.baidu.com/aistudio/projectdetail/4017905)

<a name="35"></a>
## 3.5 Customized Auto Speech Recognition and Deployment

For customized auto speech recognition and deployment, PP-ASR provides feature extraction(fbank) => Inference model（Scoring Library）=> C++ program of TLG（WFST, token, lexion, grammer). For specific usage, please refer to: [speechx](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/speechx)   
If you want to quickly use it, you can refer to [custom_streaming_asr](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/custom_streaming_asr/README_cn.md)

For more information about customized auto speech recognition and deployment, you can refer to the aistudio tutorial:
- [Customized Auto Speech Recognition](https://aistudio.baidu.com/aistudio/projectdetail/4021561)


<a name="4"></a>

## 4. Quick Start

To use PP-ASR, you can see here [install](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md), It supplies three methods to install `paddlespeech`, which are **Easy**, **Medium** and **Hard**. If you want to experience the inference function of paddlespeech, you can use **Easy** installation method.
