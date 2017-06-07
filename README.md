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
