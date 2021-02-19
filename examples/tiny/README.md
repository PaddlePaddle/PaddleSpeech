# Tiny Example

1. `source path.sh`
2. `bash run.sh`

## Steps
- Prepare the data

    ```bash
    sh local/run_data.sh
    ```

    `run_data.sh` will download dataset, generate manifests, collect normalizer's statistics and build vocabulary. Once the data preparation is done, you will find the data (only part of LibriSpeech) downloaded in `${MAIN_ROOT}/dataset/librispeech` and the corresponding manifest files generated in `${PWD}/data` as well as a mean stddev file and a vocabulary file. It has to be run for the very first time you run this dataset and is reusable for all further experiments.

- Train your own ASR model

    ```bash
    sh local/run_train.sh
    ```

    `run_train.sh` will start a training job, with training logs printed to stdout and model checkpoint of every pass/epoch saved to `${PWD}/checkpoints`. These checkpoints could be used for training resuming, inference, evaluation and deployment.

- Case inference with an existing model

    ```bash
    sh local/run_infer.sh
    ```

    `run_infer.sh` will show us some speech-to-text decoding results for several (default: 10) samples with the trained model. The performance might not be good now as the current model is only trained with a toy subset of LibriSpeech. To see the results with a better model, you can download a well-trained (trained for several days, with the complete LibriSpeech) model and do the inference:

    ```bash
    sh local/run_infer_golden.sh
    ```

- Evaluate an existing model

    ```bash
    sh local/run_test.sh
    ```

    `run_test.sh` will evaluate the model with Word Error Rate (or Character Error Rate) measurement. Similarly, you can also download a well-trained model and test its performance:

    ```bash
    sh local/run_test_golden.sh
    ```
