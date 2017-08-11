# Run DS2 on PaddleCloud

>Note: Make sure current directory is `models/deep_speech_2/cloud/`

## Step1  Configure data set

You can configure your input data and output path in pcloud_submit.sh:

-  `TRAIN_MANIFEST`： Absolute path of train data manifest file  in local file system.This file has format as bellow:

```
{"audio_filepath": "/home/disk1/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac", "duration": 5.855, "text
": "mister quilter is the ..."}
{"audio_filepath": "/home/disk1/LibriSpeech/dev-clean/1272/128104/1272-128104-0001.flac", "duration": 4.815, "text
": "nor is mister ..."}
```

- `TEST_MANIFEST`: Absolute path of train data manifest file in local filesystem.This file has format like  TRAIN_MANIFEST.

- `VOCAB_FILE`:  Absolute path of vocabulary file in local filesytem.
- `MEAN_STD_FILE`: Absolute path of vocabulary file in local filesytem.
- `CLOUD_DATA_DIR:`  Absolute path in PaddleCloud filesystem. We will upload local train data to this directory.
- `CLOUD_MODEL_DIR`: Absolute path in PaddleCloud filesystem. PaddleCloud trainer will save model to this directory.


>Note: Upload will be skipped if target file has existed in  ${CLOUD_DATA_DIR}.

## Step2  Configure computation resource

You can configure computation resource in pcloud_submit.sh:
```
# Configure computation resource and submit job to PaddleCloud
 paddlecloud submit \
 -image wanghaoshuang/pcloud_ds2:latest \
 -jobname ${JOB_NAME} \
 -cpu 4 \
 -gpu 4 \
 -memory 10Gi \
 -parallelism 1 \
 -pscpu 1 \
 -pservers 1 \
 -psmemory 10Gi \
 -passes 1 \
 -entry "sh pcloud_train.sh ${CLOUD_DATA_DIR} ${CLOUD_MODEL_DIR}" \
 ${DS2_PATH}
```
For more information, please refer to[PaddleCloud](https://github.com/PaddlePaddle/cloud/blob/develop/doc/usage_cn.md#提交任务)

## Step3  Configure algorithm options
You can configure algorithm options in pcloud_train.sh:
```
python train.py \
--use_gpu=1 \
--trainer_count=4 \
--batch_size=256 \
--mean_std_filepath=$MEAN_STD_FILE \
--train_manifest_path='./local.train.manifest' \
--dev_manifest_path='./local.test.manifest' \
--vocab_filepath=$VOCAB_PATH \
--output_model_dir=${MODEL_PATH}
```
You can get more information about algorithm options by follow command:
```
cd ..
python train.py --help
```

## Step4  Submit job
```
$ sh pcloud_submit.sh
```


## Step5 Get logs
```
$ paddlecloud logs -n 10000 deepspeech20170727130129
```
For more information, please refer to [PaddleCloud client](https://github.com/PaddlePaddle/cloud/blob/develop/doc/usage_cn.md#下载并配置paddlecloud) or get help by follow command:
```
paddlecloud --help
```
