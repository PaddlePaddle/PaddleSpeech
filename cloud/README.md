#DeepSpeech2 on paddle cloud

## Run DS2 by public data

**Step1: ** Make sure current dir is `models/deep_speech_2/cloud/`

**Step2:**  Submit job by cmd: `sh pcloud_submit.sh`

```
$ sh pcloud_submit.sh
$ uploading: deepspeech.tar.gz...
$ uploading: pcloud_prepare_data.py...
$ uploading: pcloud_split_data.py...
$ uploading: pcloud_submit.sh...
$ uploading: pcloud_train.sh...
$ deepspeech20170727130129 submited.
```
The we can get job name 'deepspeech20170727130129' at last line

**Step3:** Get logs from paddle cloud by cmd: `paddlecloud logs -n 10000 deepspeech20170727130129`.

```
$ paddlecloud logs -n 10000 deepspeech20170727130129
```
[More  options and cmd about paddle cloud](https://github.com/PaddlePaddle/cloud/blob/develop/doc/usage_cn.md)

## Run DS2 by customize data
TODO
