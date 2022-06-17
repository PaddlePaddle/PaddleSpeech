#!/bin/bash 

if [ $# != 1 ];then
    echo "usage: $0 wav_scp"
    exit -1
fi

scp=$1

# calc RTF
# wav_scp can generate from `speechx/examples/ds2_ol/aishell`

exp=exp
mkdir -p $exp

python3 local/websocket_client.py --server_ip 127.0.0.1 --port 8090 --wavscp $scp &> $exp/log.rsl

python3 local/rtf_from_log.py --logfile $exp/log.rsl


 