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
$ ==========================deepspeech20170727130129-trainer-6vk3m==========================
label selector: paddle-job-pserver=deepspeech20170727130129, desired: 1
running pod list:  [('Running', '10.1.3.6')]
label selector: paddle-job=deepspeech20170727130129, desired: 1
running pod list:  [('Running', '10.1.83.14')]
Starting training job:  /pfs/dlnel/home/yanxu05@baidu.com/jobs/deepspeech20170727130129, num_gradient_servers: 1, trainer_id:  0, version:  v2
I0727 05:01:42.969719    25 Util.cpp:166] commandline:  --num_gradient_servers=1 --ports_num_for_sparse=1 --use_gpu=1 --trainer_id=0 --pservers=10.1.3.6 --trainer_count=4 --num_passes=1 --ports_num=1 --port=7164
[INFO 2017-07-27 05:01:50,279 layers.py:2430] output for __conv_0__: c = 32, h = 81, w = 54, size = 139968
[WARNING 2017-07-27 05:01:50,280 layers.py:2789] brelu is not recommend for batch normalization's activation, maybe the relu is better
[INFO 2017-07-27 05:01:50,283 layers.py:2430] output for __conv_1__: c = 32, h = 41, w = 54, size = 70848
[WARNING 2017-07-27 05:01:50,283 layers.py:2789] brelu is not recommend for batch normalization's activation, maybe the relu is better
[WARNING 2017-07-27 05:01:50,287 layers.py:2789]  is not recommend for batch normalization's activation, maybe the relu is better
[WARNING 2017-07-27 05:01:50,291 layers.py:2789]  is not recommend for batch normalization's activation, maybe the relu is better
[WARNING 2017-07-27 05:01:50,295 layers.py:2789]  is not recommend for batch normalization's activation, maybe the relu is better
I0727 05:01:50.316176    25 MultiGradientMachine.cpp:99] numLogicalDevices=1 numThreads=4 numDevices=4
I0727 05:01:50.454787    25 GradientMachine.cpp:85] Initing parameters..
I0727 05:01:50.690007    25 GradientMachine.cpp:92] Init parameters done.
```
[More  optins and cmd aoubt paddle cloud](https://github.com/PaddlePaddle/cloud/blob/develop/doc/usage_cn.md)

## Run DS2 by customize data
TODO
