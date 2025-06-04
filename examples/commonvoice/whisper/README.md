# Whisper Fine-tuning on Common Voice

This example demonstrates how to fine-tune Whisper models on the [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) dataset using PaddleSpeech.

## Overview

Whisper is a state-of-the-art speech recognition model from OpenAI. This implementation allows you to fine-tune Whisper models on new datasets to improve performance for specific languages, domains, or accents.

## Features

- Complete fine-tuning pipeline for Whisper models on custom datasets
- Flexible configuration via YAML files
- Support for all Whisper model sizes (tiny, base, small, medium, large, etc.)
- Data preparation tools for Common Voice and custom datasets
- Distributed training support with mixed precision
- Gradient accumulation for large batch training
- Learning rate scheduling and optimization techniques
- Evaluation tools with WER/CER metrics
- Command-line inference with both fine-tuned and original Whisper models
- Model export utilities for deployment
- Visualization tools for performance analysis

## Installation

Ensure you have PaddleSpeech installed with all dependencies:

```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
pip install -e .
pip install datasets soundfile librosa matplotlib pandas jiwer
```

## Data

We use the Common Voice dataset (version 11.0) available on Hugging Face:
https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0

Other datasets compatible with the pipeline include LibriSpeech, AISHELL, and any dataset that can be converted to the manifest format (see below).

## Unified Command-Line Interface

This example includes a unified CLI for all operations:

```bash
python whisper_cli.py COMMAND [OPTIONS]
```

Available commands:
- `prepare`: Prepare dataset for fine-tuning
- `train`: Fine-tune Whisper model
- `evaluate`: Evaluate model performance
- `infer`: Run inference with fine-tuned or original model

## Data Preparation

To download and prepare the Common Voice dataset:

```bash
python whisper_cli.py prepare --language en --output_dir ./data
```

Options:
- `--language`: Target language code (default: en)
- `--output_dir`: Directory to save preprocessed data (default: ./data)
- `--cache_dir`: Cache directory for HuggingFace datasets
- `--val_size`: Validation set size ratio (default: 0.03)
- `--test_size`: Test set size ratio (default: 0.03)
- `--min_duration`: Minimum audio duration in seconds (default: 0.5)
- `--max_duration`: Maximum audio duration in seconds (default: 30.0)

Manifest format:
```json
{"audio": "path/to/audio.wav", "text": "transcription", "duration": 3.45}
```

## Configuration

Fine-tuning parameters are specified in YAML config files. See `conf/whisper_base.yaml` for a detailed example.

Key configuration sections:
- **model**: Model size, checkpoint path, freeze options
- **data**: Dataset paths, languages, tasks
- **training**: Batch size, learning rate, optimizer settings
- **distributed**: Distributed training options
- **output**: Save paths, logging options

## Training

To fine-tune the Whisper model:

```bash
python whisper_cli.py train --config conf/whisper_base.yaml --resource_path ./resources
```

For distributed training:

```bash
python -m paddle.distributed.launch --gpus "0,1,2,3" whisper_cli.py train --config conf/whisper_base.yaml --distributed True
```

Options:
- `--config`: Path to configuration YAML file
- `--resource_path`: Path to resources directory containing model assets
- `--device`: Device to use (cpu, gpu, xpu)
- `--seed`: Random seed
- `--checkpoint_path`: Path to resume training from checkpoint
- `--distributed`: Enable distributed training

## Evaluation

Evaluate model performance on a test set:

```bash
python whisper_cli.py evaluate --manifest ./data/test_manifest.json --checkpoint ./exp/whisper_fine_tune/epoch_10 --output_dir ./eval_results
```

Options:
- `--manifest`: Path to test manifest file
- `--checkpoint`: Path to model checkpoint
- `--model_size`: Model size if using original Whisper
- `--language`: Language code
- `--output_dir`: Directory to save evaluation results
- `--max_samples`: Maximum number of samples to evaluate

## Inference

For transcribing audio with a fine-tuned model:

```bash
python whisper_cli.py infer --audio_file path/to/audio.wav --checkpoint ./exp/whisper_fine_tune/final
```

For batch processing a directory:

```bash
python whisper_cli.py infer --audio_dir path/to/audio/folder --output_dir ./transcriptions --checkpoint ./exp/whisper_fine_tune/final
```

For inference with the original Whisper models:

```bash
python whisper_cli.py infer --audio_file path/to/audio.wav --use_original --model_size large-v3 --resource_path ./resources
```

Options:
- `--audio_file`: Path to single audio file
- `--audio_dir`: Path to directory with audio files
- `--checkpoint`: Path to fine-tuned checkpoint
- `--use_original`: Use original Whisper model
- `--model_size`: Model size (tiny, base, small, medium, large, etc.)
- `--language`: Language code (or "auto" for detection)
- `--task`: Task type (transcribe or translate)
- `--beam_size`: Beam size for beam search
- `--temperature`: Temperature for sampling
- `--without_timestamps`: Don't include timestamps

## Visualization

Visualize evaluation results:

```bash
python visualize.py --results_file ./eval_results/evaluation_results.json --output_dir ./visualizations
```

Options:
- `--results_file`: Path to evaluation results JSON file
- `--output_dir`: Directory to save visualizations
- `--audio_dir`: Directory with audio files (optional)
- `--num_samples`: Number of individual samples to visualize
- `--show`: Show plots interactively

## Model Export

Export fine-tuned model to inference format:

```bash
python export_model.py --checkpoint ./exp/whisper_fine_tune/final --output_path ./exported_model --model_size base
```

Options:
- `--checkpoint`: Path to model checkpoint
- `--output_path`: Path to save exported model
- `--model_size`: Model size

## Advanced Usage

### Freezing Encoder

To freeze the encoder and only fine-tune the decoder, set the following in your config file:

```yaml
model:
  freeze_encoder: true
```

### Gradient Accumulation

For effective training with limited GPU memory, use gradient accumulation:

```yaml
training:
  accum_grad: 8  # Accumulate gradients over 8 batches
```

### Mixed Precision

Enable mixed precision training for faster computation:

```yaml
training:
  amp: true  # Enable automatic mixed precision
```

### Custom Datasets

To use custom datasets, prepare manifest files in the following format:

```json
{"audio": "/absolute/path/to/audio.wav", "text": "transcription text"}
```

Then specify the manifest paths in your config file:

```yaml
data:
  train_manifest: path/to/train_manifest.json
  dev_manifest: path/to/dev_manifest.json
  test_manifest: path/to/test_manifest.json
```

## Reference

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Fine-tuning Whisper](https://huggingface.co/blog/fine-tune-whisper)
- [Paper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/pdf/2212.04356)
- [Common Voice Dataset](https://commonvoice.mozilla.org/)
- [PaddleSpeech Documentation](https://github.com/PaddlePaddle/PaddleSpeech)
