# Data Preparation

## Generate Manifest

*DeepSpeech2 on PaddlePaddle* accepts a textual **manifest** file as its data set interface. A manifest file summarizes a set of speech data, with each line containing some meta data (e.g. file path, transcription, duration) of one audio clip, in [JSON](http://www.json.org/) format, such as:

```
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0001.flac", "duration": 3.275, "text": "stuff it into you his belly counselled him"}
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0007.flac", "duration": 4.275, "text": "a cold lucid indifference reigned in his soul"}
```
To use your custom data, you only need to generate such manifest files to summarize the dataset. Given such summarized manifests, training, inference and all other modules can be aware of where to access the audio files, as well as their meta data including the transcription labels.

For how to generate such manifest files, please refer to `examples/librispeech/local/librispeech.py`, which will download data and generate manifest files for LibriSpeech dataset.

## Compute Mean & Stddev for Normalizer

To perform z-score normalization (zero-mean, unit stddev) upon audio features, we have to estimate in advance the mean and standard deviation of the features, with some training samples:

```bash
python3 utils/compute_mean_std.py \
--num_samples 2000 \
--spectrum_type linear \
--manifest_path examples/librispeech/data/manifest.train \
--output_path examples/librispeech/data/mean_std.npz
```

It will compute the mean and standard deviations of the power spectrum feature with 2000 random sampled audio clips listed in `examples/librispeech/data/manifest.train` and save the results to `examples/librispeech/data/mean_std.npz` for further usage.


## Build Vocabulary

A vocabulary of possible characters is required to convert the transcription into a list of token indices for training, and in decoding, to convert from a list of indices back to the text again. Such a character-based vocabulary can be built with `utils/build_vocab.py`.

```bash
python3 utils/build_vocab.py \
--count_threshold 0 \
--vocab_path examples/librispeech/data/eng_vocab.txt \
--manifest_paths examples/librispeech/data/manifest.train
```

It will write a vocabulary file `examples/librispeech/data/vocab.txt` with all transcription text in `examples/librispeech/data/manifest.train`, without vocabulary truncation (`--count_threshold 0`).
