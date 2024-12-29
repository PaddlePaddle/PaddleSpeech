python -m pip install -r ../audiotools/requirements.txt
# wget -P ./test_data https://paddlespeech.bj.bcebos.com/datasets/unit_test/asr/static_ds2online_inputs.pickle
# wget 
find . -name "*✅.py" | xargs python -m pytest