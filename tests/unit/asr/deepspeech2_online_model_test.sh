mkdir -p ./test_data
wget -P ./test_data https://paddlespeech.bj.bcebos.com/datasets/unit_test/asr/static_ds2online_inputs.pickle
python deepspeech2_online_model_test.py
