rm exp -rf
rm e2e -rf
bash prepare.sh whole_train_infer
bash test.sh speedyspeech_params_whole_single_gpu.txt whole_train_infer
rm exp -rf
rm e2e -rf
bash test.sh speedyspeech_params_whole_multi_gpu.txt whole_train_infer
