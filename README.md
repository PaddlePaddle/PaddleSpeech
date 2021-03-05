# DeepSpeech on PaddlePaddle

[中文版](README_cn.md)

*DeepSpeech on PaddlePaddle* is an open-source implementation of end-to-end Automatic Speech Recognition (ASR) engine, with [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform. Our vision is to empower both industrial application and academic research on speech recognition, via an easy-to-use, efficient and scalable implementation, including training, inference & testing module, and demo deployment.

For more information, please docs under `doc`.

## Models
* [Baidu's Deep Speech2](http://proceedings.mlr.press/v48/amodei16.pdf)

## Setup
* python3.7
* paddlepaddle 2.0.0

- Run the setup script for the remaining dependencies

```bash
git clone https://github.com/PaddlePaddle/DeepSpeech.git
cd DeepSpeech
pushd tools; make; popd
source tools/venv/bin/activate
bash setup.sh
```

- Source venv before do experiment.

```bash
source tools/venv/bin/activate
```

## Getting Started

Please see [Getting Started](docs/geting_started.md) and [tiny egs](examples/tiny/README.md).

## Questions and Help

You are welcome to submit questions and bug reports in [Github Issues](https://github.com/PaddlePaddle/DeepSpeech/issues). You are also welcome to contribute to this project.
