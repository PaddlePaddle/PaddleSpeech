export CUDA_VISIBLE_DEVICE=0,1,2,3
# we need the streaming asr server
nohup python3 streaming_asr_server.py --config_file conf/ws_conformer_application.yaml > streaming_asr.log  2>&1  &

# start the acs server
nohup paddlespeech_server start --config_file conf/acs_application.yaml > acs.log 2>&1 &

