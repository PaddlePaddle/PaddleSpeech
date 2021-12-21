# Quick Start of Audio Classification
Several shell scripts provided in `./examples/esc50/cls0` will help us to quickly give it a try, for most major modules, including data preparation, model training, model evaluation, with [ESC50](ttps://github.com/karolpiczak/ESC-50) dataset.

Some of the scripts in `./examples` are not configured with GPUs. If you want to train with 8 GPUs, please modify `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. If you don't have any GPU available, please set `CUDA_VISIBLE_DEVICES=` to use CPUs instead.

Let's start a audio classification task with the following steps:

- Go to the directory

    ```bash
    cd examples/esc50/cls0
    ```

- Source env
    ```bash
    source path.sh
    ```

- Main entry point
    ```bash
    CUDA_VISIBLE_DEVICES=0 ./run.sh 1
    ```

This demo includes fine-tuning, evaluating and deploying a audio classificatio model. More detailed information is provided in the following sections. 

## Fine-tuning a model
PANNs([PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211.pdf)) are pretrained models with [Audioset](https://research.google.com/audioset/). They can be easily used to extract audio embeddings for audio classification task.

To start a model fine-tuning, please run:
```bash
ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
feat_backend=numpy
./local/train.sh ${ngpu} ${feat_backend}
```

## Deploy a model
Once you save a model checkpoint, you can export it to static graph and deploy by python scirpt:

- Export to a static graph
    ```bash
    ./local/export.sh ${ckpt_dir} ./export
    ```
    The argument `ckpt_dir` should be a directory in which a model checkpoint stored, for example `checkpoint/epoch_50`.

    The static graph will be exported to `./export`.

- Inference
    ```bash
    ./local/static_model_infer.sh ${infer_device} ./export ${audio_file}
    ```
    The argument `infer_device` can be `cpu` or `gpu`, and it means which device to be used to infer. And `audio_file` should be a wave file with name `*.wav`.
