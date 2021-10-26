rm exp -rf
rm e2e -rf
bash prepare.sh lite_train_infer
bash test.sh speedyspeech_params_lite_single_gpu.txt lite_train_infer
rm exp -rf
rm e2e -rf
bash test.sh speedyspeech_params_lite_multi_gpu.txt lite_train_infer
