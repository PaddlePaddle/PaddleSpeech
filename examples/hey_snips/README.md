# MDTC Keyword Spotting with HeySnips Dataset

## Dataset

Before running scripts, you **MUST** follow this instruction to download the dataset: https://github.com/sonos/keyword-spotting-research-datasets

After you download and decompress the dataset archive, you should **REPLACE** the value of `data_dir` in `conf/*.yaml` to complete dataset config.

## Get Started

In this section, we will train the [MDTC](https://arxiv.org/pdf/2102.13552.pdf) model and evaluate on "Hey Snips" dataset.

```sh
CUDA_VISIBLE_DEVICES=0,1 ./run.sh conf/mdtc.yaml
```

This script contains training and scoring steps. You can just set the `CUDA_VISIBLE_DEVICES` environment var to run on single gpu or multi-gpus.

The vars `stage` and `stop_stage` in `./run.sh` controls the running steps:
- stage 1: Training from scratch.
- stage 2: Evaluating model on test dataset and computing detection error tradeoff(DET) of all trigger thresholds.
- stage 3: Plotting the DET cruve for visualizaiton.
