DS2_PATH=../
tar -czf deepspeech.tar.gz ${DS2_PATH}
JOB_NAME=deepspeech`date +%Y%m%d%H%M%S`
cp pcloud_train.sh ${DS2_PATH}
paddlecloud submit \
-image wanghaoshuang/pcloud_ds2:latest-gpu-cudnn \
-jobname ${JOB_NAME} \
-cpu 4 \
-gpu 4 \
-memory 10Gi \
-parallelism 1 \
-pscpu 1 \
-pservers 1 \
-psmemory 10Gi \
-passes 1 \
-entry "sh pcloud_train.sh" \
.
