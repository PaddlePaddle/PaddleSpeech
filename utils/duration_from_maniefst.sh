#!/bin/bash

script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x $script_dir/../paddle_log
$script_dir/../paddle_log

if [ $# == 1 ];then
    echo "usage: ${0} manifest_file"
    exit -1
fi

manifest=$1

jq -S '.feat_shape[0]' ${manifest} | sort -nu
