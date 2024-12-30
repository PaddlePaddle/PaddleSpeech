# Parallel WaveGAN with AISHELL-3
This example contains code used to train a [parallel wavegan](http://arxiv.org/abs/1910.11480) model with [AISHELL-3](http://www.aishelltech.com/aishell_3).

AISHELL-3 is a large-scale and high-fidelity multi-speaker Mandarin speech corpus that could be used to train multi-speaker Text-to-Speech (TTS) systems.
## Dataset
### Download and Extract
Download AISHELL-3 from it's [Official Website](http://www.aishelltech.com/aishell_3) and extract it to `~/datasets`. Then the dataset is in the directory `~/datasets/data_aishell3`.

### Get MFA Result and Extract
We use [MFA2.x](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for aishell3_fastspeech2.
You can download from here [aishell3_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/aishell3_alignment_tone.tar.gz), or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) (use MFA1.x now) of our repo.

## Get Started
Assume the path to the dataset is `~/datasets/data_aishell3`.
Assume the path to the MFA result of AISHELL-3 is `./aishell3_alignment_tone`.
Run the command below to
1. **source path**.
2. preprocess the dataset.
3. train the model.
4. synthesize wavs.
    - synthesize waveform from `metadata.jsonl`.
```bash
./run.sh
```
You can choose a range of stages you want to run, or set `stage` equal to `stop-stage` to use only one stage, for example, run the following command will only preprocess the dataset.
```bash
./run.sh --stage 0 --stop-stage 0
```
### Data Preprocessing
```bash
./local/preprocess.sh ${conf_path}
```
When it is done. A `dump` folder is created in the current directory. The structure of the dump folder is listed below.

```text
dump
├── dev
│   ├── norm
│   └── raw
├── test
│   ├── norm
│   └── raw
└── train
    ├── norm
    ├── raw
    └── feats_stats.npy
```

The dataset is split into 3 parts, namely `train`, `dev`, and `test`, each of which contains a `norm` and `raw` subfolder. The `raw` folder contains the log magnitude of the mel spectrogram of each utterance, while the norm folder contains the normalized spectrogram. The statistics used to normalize the spectrogram are computed from the training set, which is located in `dump/train/feats_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains id and paths to the spectrogram of each utterance.

### Model Training
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
`./local/train.sh` calls `${BIN_DIR}/train.py`.
Here's the complete help message.

```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--ngpu NGPU] [--batch-size BATCH_SIZE] [--max-iter MAX_ITER]
                [--run-benchmark RUN_BENCHMARK]
                [--profiler_options PROFILER_OPTIONS]

Train a ParallelWaveGAN model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       ParallelWaveGAN config file.
  --train-metadata TRAIN_METADATA
                        training data.
  --dev-metadata DEV_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --ngpu NGPU           if ngpu == 0, use cpu.

benchmark:
  arguments related to benchmark.

  --batch-size BATCH_SIZE
                        batch size.
  --max-iter MAX_ITER   train max steps.
  --run-benchmark RUN_BENCHMARK
                        runing benchmark or not, if True, use the --batch-size
                        and --max-iter.
  --profiler_options PROFILER_OPTIONS
                        The option of profiler, which should be in format
                        "key1=value1;key2=value2;key3=value3".
```

1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are saved in `checkpoints/` inside this directory.
4. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

### Synthesizing
`./local/synthesize.sh` calls `${BIN_DIR}/../synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h] [--generator-type GENERATOR_TYPE] [--config CONFIG]
                     [--checkpoint CHECKPOINT] [--test-metadata TEST_METADATA]
                     [--output-dir OUTPUT_DIR] [--ngpu NGPU]

Synthesize with GANVocoder.

optional arguments:
  -h, --help            show this help message and exit
  --generator-type GENERATOR_TYPE
                        type of GANVocoder, should in {pwgan, mb_melgan,
                        style_melgan, } now
  --config CONFIG       GANVocoder config file.
  --checkpoint CHECKPOINT
                        snapshot to load.
  --test-metadata TEST_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --ngpu NGPU           if ngpu == 0, use cpu.
```

1. `--config` parallel wavegan config file. You should use the same config with which the model is trained.
2. `--checkpoint` is the checkpoint to load. Pick one of the checkpoints from `checkpoints` inside the training output directory. If you use the pretrained model, use the `snapshot_iter_1000000.pdz `.
3. `--test-metadata` is the metadata of the test dataset. Use the `metadata.jsonl` in the `dev/norm` subfolder from the processed directory.
4. `--output-dir` is the directory to save the synthesized audio files.
5. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

## Pretrained Models
Pretrained models can be downloaded here:
- [pwg_aishell3_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip)

The static model can be downloaded here:
- [pwgan_aishell3_static_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_aishell3_static_1.1.0.zip)

The PIR static model can be downloaded here:
- [pwgan_aishell3_static_pir_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_aishell3_static_pir_1.1.0.zip) (Run PIR model need to set FLAGS_enable_pir_api=1, and PIR model only worked with paddlepaddle>=3.0.0b2)

The ONNX model can be downloaded here:
- [pwgan_aishell3_onnx_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_aishell3_onnx_1.1.0.zip)

The Paddle-Lite model can be downloaded here:
- [pwgan_aishell3_pdlite_1.3.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_aishell3_pdlite_1.3.0.zip)

Model | Step | eval/generator_loss | eval/log_stft_magnitude_loss:| eval/spectral_convergence_loss 
:-------------:| :------------:| :-----: | :-----: | :--------:
default| 1(gpu) x 400000|1.968762|0.759008|0.218524

Parallel WaveGAN checkpoint contains files listed below.

```text
pwg_aishell3_ckpt_0.5
├── default.yaml                   # default config used to train parallel wavegan
├── feats_stats.npy                # statistics used to normalize spectrogram when training parallel wavegan
└── snapshot_iter_1000000.pdz      # generator parameters of parallel wavegan
```
## Acknowledgement
We adapted some code from https://github.com/kan-bayashi/ParallelWaveGAN.
