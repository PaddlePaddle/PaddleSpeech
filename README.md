# Deep Speech 2 on PaddlePaddle

## Installation

Please replace `$PADDLE_INSTALL_DIR` with your own paddle installation directory.

```
pip install -r requirements.txt
export LD_LIBRARY_PATH=$PADDLE_INSTALL_DIR/Paddle/third_party/install/warpctc/lib:$LD_LIBRARY_PATH
```

For some machines, we also need to install libsndfile1. Details to be added.

## Usage

### Preparing Data

```
cd data
python librispeech.py
cat manifest.libri.train-* > manifest.libri.train-all
cd ..
```

After running librispeech.py, we have several "manifest" json files named with a prefix `manifest.libri.`. A manifest file summarizes a speech data set, with each line containing the meta data (i.e. audio filepath, transcription text, audio duration) of each audio file within the data set, in json format.

By `cat manifest.libri.train-* > manifest.libri.train-all`, we simply merge the three seperate sample sets of LibriSpeech (train-clean-100, train-clean-360, train-other-500) into one training set. This is a simple way for merging different data sets.

More help for arguments:

```
python librispeech.py --help
```

### Traininig

For GPU Training:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --trainer_count 4 --train_manifest_path ./data/manifest.libri.train-all
```

For CPU Training:

```
python train.py --trainer_count 8 --use_gpu False -- train_manifest_path ./data/manifest.libri.train-all
```

More help for arguments:

```
python train.py --help
```

### Inferencing

```
python infer.py
```

More help for arguments:

```
python infer.py --help
```
