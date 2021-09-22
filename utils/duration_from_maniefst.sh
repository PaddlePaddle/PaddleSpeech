#!/bin/bash

if [ $# == 1 ];then
    echo "usage: ${0} manifest_file"
    exit -1
fi

manifest=$1

jq -S '.feat_shape[0]' ${manifest} | sort -nu
