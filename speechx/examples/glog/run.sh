#!/bin/bash
set +x
set -e

. ./path.sh

# 1. compile
if [ ! -d ${SPEECHX_EXAMPLES} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh
    popd
fi

# 2. run 
glog_test

echo "------"
export FLAGS_logtostderr=1 
glog_test

echo "------"
glog_logtostderr_test
