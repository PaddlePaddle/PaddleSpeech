([简体中文](./README_cn.md)|English)
# Speech Verification)

## Introduction

Speaker Verification, refers to the problem of getting a speaker embedding from an audio. 

This demo is an implementation to extract speaker embedding from a specific audio file. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.

### 2. Prepare Input File
The input of this demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for this demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
```

### 3. Usage
- Command Line(Recommended)
  ```bash
  paddlespeech vector --task spk --input 85236145389.wav

  echo -e "demo1 85236145389.wav" > vec.job
  paddlespeech vector --task spk --input vec.job

  echo -e "demo2 85236145389.wav \n demo3 85236145389.wav" | paddlespeech vector --task spk
  ```
  
  Usage:
  ```bash
  paddlespeech vector --help
  ```
  Arguments:
  - `input`(required): Audio file to recognize.
  - `task` (required): Specify `vector` task. Default `spk`。
  - `model`: Model type of vector task. Default: `ecapatdnn_voxceleb12`.
  - `sample_rate`: Sample rate of the model. Default: `16000`.
  - `config`: Config of vector task. Use pretrained model when it is None. Default: `None`.
  - `ckpt_path`: Model checkpoint. Use pretrained model when it is None. Default: `None`.
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.

  Output:

  ```bash
    demo [  1.4217498    5.626253    -5.342073     1.1773866    3.308055
    1.756596     5.167894    10.80636     -3.8226728   -5.6141334
    2.623845    -0.8072968    1.9635103   -7.3128724    0.01103897
    -9.723131     0.6619743   -6.976803    10.213478     7.494748
    2.9105635    3.8949256    3.7999806    7.1061673   16.905321
    -7.1493764    8.733103     3.4230042   -4.831653   -11.403367
    11.232214     7.1274667   -4.2828417    2.452362    -5.130748
    -18.177666    -2.6116815  -11.000337    -6.7314315    1.6564683
    0.7618269    1.1253023   -2.083836     4.725744    -8.782597
    -3.539873     3.814236     5.1420674    2.162061     4.096431
    -6.4162116   12.747448     1.9429878  -15.152943     6.417416
    16.097002    -9.716668    -1.9920526   -3.3649497   -1.871939
    11.567354     3.69788     11.258265     7.442363     9.183411
    4.5281515   -1.2417862    4.3959084    6.6727695    5.8898783
    7.627124    -0.66919386 -11.889693    -9.208865    -7.4274073
    -3.7776625    6.917234    -9.848748    -2.0944717   -5.135116
    0.49563864   9.317534    -5.9141874   -1.8098574   -0.11738578
    -7.169265    -1.0578263   -5.7216787   -5.1173844   16.137651
    -4.473626     7.6624317   -0.55381083   9.631587    -6.4704556
    -8.548508     4.3716145   -0.79702514   4.478997    -2.9758704
    3.272176     2.8382776    5.134597    -9.190781    -0.5657382
    -4.8745747    2.3165567   -5.984303    -2.1798875    0.35541576
    -0.31784213   9.493548     2.1144536    4.358092   -12.089823
    8.451689    -7.925461     4.6242585    4.4289427   18.692003
    -2.6204622   -5.149185    -0.35821092   8.488551     4.981496
    -9.32683     -2.2544234    6.6417594    1.2119585   10.977129
    16.555033     3.3238444    9.551863    -1.6676947   -0.79539716
    -8.605674    -0.47356385   2.6741948   -5.359179    -2.6673796
    0.66607     15.443222     4.740594    -3.4725387   11.592567
    -2.054497     1.7361217   -8.265324    -9.30447      5.4068313
    -1.5180256   -7.746615    -6.089606     0.07112726  -0.34904733
    -8.649895    -9.998958    -2.564841    -0.53999114   2.601808
    -0.31927416  -1.8815292   -2.07215     -3.4105783   -8.2998085
    1.483641   -15.365992    -8.288208     3.8847756   -3.4876456
    7.3629923    0.4657332    3.132599    12.438889    -1.8337058
    4.532936     2.7264361   10.145339    -6.521951     2.897153
    -3.3925855    5.079156     7.759716     4.677565     5.8457737
    2.402413     7.7071047    3.9711342   -6.390043     6.1268735
    -3.7760346  -11.118123  ]
  ```

- Python API
  ```python
  import paddle
  from paddlespeech.cli import VectorExecutor

  vector_executor = VectorExecutor()
  audio_emb = vector_executor(
      model='ecapatdnn_voxceleb12',
      sample_rate=16000,
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      ckpt_path=None,
      audio_file='./85236145389.wav',
      device=paddle.get_device())
  print('Audio embedding Result: \n{}'.format(audio_emb))
  ```

  Output：

  ```bash
  # Vector Result:
   Audio embedding Result:
    [  1.4217498    5.626253    -5.342073     1.1773866    3.308055
    1.756596     5.167894    10.80636     -3.8226728   -5.6141334
    2.623845    -0.8072968    1.9635103   -7.3128724    0.01103897
    -9.723131     0.6619743   -6.976803    10.213478     7.494748
    2.9105635    3.8949256    3.7999806    7.1061673   16.905321
    -7.1493764    8.733103     3.4230042   -4.831653   -11.403367
    11.232214     7.1274667   -4.2828417    2.452362    -5.130748
    -18.177666    -2.6116815  -11.000337    -6.7314315    1.6564683
    0.7618269    1.1253023   -2.083836     4.725744    -8.782597
    -3.539873     3.814236     5.1420674    2.162061     4.096431
    -6.4162116   12.747448     1.9429878  -15.152943     6.417416
    16.097002    -9.716668    -1.9920526   -3.3649497   -1.871939
    11.567354     3.69788     11.258265     7.442363     9.183411
    4.5281515   -1.2417862    4.3959084    6.6727695    5.8898783
    7.627124    -0.66919386 -11.889693    -9.208865    -7.4274073
    -3.7776625    6.917234    -9.848748    -2.0944717   -5.135116
    0.49563864   9.317534    -5.9141874   -1.8098574   -0.11738578
    -7.169265    -1.0578263   -5.7216787   -5.1173844   16.137651
    -4.473626     7.6624317   -0.55381083   9.631587    -6.4704556
    -8.548508     4.3716145   -0.79702514   4.478997    -2.9758704
    3.272176     2.8382776    5.134597    -9.190781    -0.5657382
    -4.8745747    2.3165567   -5.984303    -2.1798875    0.35541576
    -0.31784213   9.493548     2.1144536    4.358092   -12.089823
    8.451689    -7.925461     4.6242585    4.4289427   18.692003
    -2.6204622   -5.149185    -0.35821092   8.488551     4.981496
    -9.32683     -2.2544234    6.6417594    1.2119585   10.977129
    16.555033     3.3238444    9.551863    -1.6676947   -0.79539716
    -8.605674    -0.47356385   2.6741948   -5.359179    -2.6673796
    0.66607     15.443222     4.740594    -3.4725387   11.592567
    -2.054497     1.7361217   -8.265324    -9.30447      5.4068313
    -1.5180256   -7.746615    -6.089606     0.07112726  -0.34904733
    -8.649895    -9.998958    -2.564841    -0.53999114   2.601808
    -0.31927416  -1.8815292   -2.07215     -3.4105783   -8.2998085
    1.483641   -15.365992    -8.288208     3.8847756   -3.4876456
    7.3629923    0.4657332    3.132599    12.438889    -1.8337058
    4.532936     2.7264361   10.145339    -6.521951     2.897153
    -3.3925855    5.079156     7.759716     4.677565     5.8457737
    2.402413     7.7071047    3.9711342   -6.390043     6.1268735
    -3.7760346  -11.118123  ]
  ```

### 4.Pretrained Models

Here is a list of pretrained models released by PaddleSpeech that can be used by command and python API:

| Model | Sample Rate
| :--- | :---: |
| ecapatdnn_voxceleb12 | 16k
