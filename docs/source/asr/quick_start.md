# Quick Start of Speech-to-Text
Several shell scripts provided in `./examples/tiny/local` will help us to quickly give it a try, for most major modules, including data preparation, model training, case inference, and model evaluation, with a few public datasets (e.g. [LibriSpeech](http://www.openslr.org/12/), [Aishell](http://www.openslr.org/33)). Reading these examples will also help you to understand how to make it work with your data.

Some of the scripts in `./examples` are not configured with GPUs. If you want to train with 8 GPUs, please modify `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. If you don't have any GPU available, please set `CUDA_VISIBLE_DEVICES=` to use CPUs instead. Besides, if an out-of-memory problem occurs, just reduce `batch_size` to fit.

Let's take a tiny sampled subset of [LibriSpeech dataset](http://www.openslr.org/12/) for instance.

- Go to the directory

    ```bash
    cd examples/tiny
    ```
    Notice that this is only a toy example with a tiny sampled subset of LibriSpeech. If you would like to try with the complete dataset (would take several days for training), please go to `examples/librispeech` instead.
- Source env
    ```bash
    source path.sh
    ```
    **Must do this before you start to do anything.**
    Set `MAIN_ROOT` as project dir. Using the default `deepspeech2` model as `MODEL`, you can change this in the script.
- Main entry point
    ```bash
    bash run.sh
    ```
    This is just a demo, please make sure every `step` works well before the next `step`.

More detailed information is provided in the following sections. Wish you a happy journey with the *DeepSpeech on PaddlePaddle* ASR engine!

## Training a model
The key steps of training for the Mandarin language are the same as that of the English language and we have also provided an example for Mandarin training with Aishell in `examples/aishell/local`. As mentioned above, please execute `sh data.sh`, `sh train.sh` and `sh test.sh` to do data preparation, training, and testing correspondingly.

## Evaluate a Model
To evaluate a model's performance quantitatively, please run:
```bash
CUDA_VISIBLE_DEVICES=0 bash local/test.sh
```
The error rate (default: word error rate; can be set with `error_rate_type`) will be printed.

We provide two types of CTC decoders: *CTC greedy decoder* and *CTC beam search decoder*. The *CTC greedy decoder* is an implementation of the simple best-path decoding algorithm, selecting at each timestep the most likely token, thus being greedy and locally optimal. The [*CTC beam search decoder*](https://arxiv.org/abs/1408.2873) otherwise utilizes a heuristic breadth-first graph search for reaching near-global optimality; it also requires a pre-trained KenLM language model for better scoring and ranking. The decoder type can be set with the argument `decoding_method`.
