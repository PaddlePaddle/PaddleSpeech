paddlecloud submit \
-image wanghaoshuang/pcloud_ds2 \
-jobname ds23 \
-cpu 1 \
-gpu 0 \
-memory 10Gi \
-parallelism 1 \
-pscpu 1 \
-pservers 1 \
-psmemory 10Gi \
-passes 1 \
-entry "sh pcloud_train.sh" \
./deep_speech_2
