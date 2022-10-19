# Rhythm Prediction with CSMSC and AiShell3

## Get Started
### Data Preprocessing
```bash
./run.sh --stage 0 --stop-stage 0
```
### Model Training
```bash
./run.sh --stage 1 --stop-stage 1
```
### Testing
```bash
./run.sh --stage 2 --stop-stage 2
```
### Punctuation Restoration
```bash
./run.sh --stage 3 --stop-stage 3
```
## Pretrained Model
The pretrained model can be downloaded here:

[snapshot_iter_2600.pdz](https://paddlespeech.bj.bcebos.com/Rhy_Prediction/snapshot_iter_2600.pdz)

And you should put it into `exp/YOUREXP/checkpoints` folder.
