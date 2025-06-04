# Descript Audio Codec (DAC) Implementation in PaddleSpeech

This is an implementation of the Descript Audio Codec (DAC) in PaddleSpeech, based on the paper ["DAC: A Unified Approach to Neural Codec Modeling"](https://arxiv.org/abs/2306.06546).

## Overview

DAC is a neural audio codec that provides high-quality audio compression at various bit rates while maintaining excellent perceptual quality. This implementation includes:

- DAC model architecture
- Distributed training pipeline
- Inference API
- Evaluation metrics

## Features

- High-quality audio compression and reconstruction
- Variable bitrate support
- Support for different audio domains
- Compatible with PaddleSpeech's distributed training infrastructure

## Usage

### Training

See the example training script at `examples/audio/codec/dac/train.py`

### Inference

See the example inference script at `examples/audio/codec/dac/infer.py`

### Evaluation

See the example evaluation script at `examples/audio/codec/dac/evaluate.py`

## Citation

```bibtex
@article{kumar2023dac,
  title={DAC: A Unified Approach to Neural Codec Modeling},
  author={Kumar, Manoj and Shor, Joel and Zhang, Yu and Han, Wei and Wu, Yonghui and Itkin, Eli and Venkatesh, Sree and Ryabov, Andrew and Rubanov, Oleg and Wilkins, Peter and Chen, Jiayu and Olga Vitek, Bryan Catanzaro},
  journal={arXiv preprint arXiv:2306.06546},
  year={2023}
}
```
