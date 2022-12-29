#!/bin/bash
 
now=`date +'%Y-%m-%d %H:%M:%S'`
start_time=$(date --date="$now" +%s);

compute_my_fbank_main_kaldi_native_fbank scp:aishell_test.scp 
 
now=`date +'%Y-%m-%d %H:%M:%S'`
end_time=$(date --date="$now" +%s);
echo "openblas fbank used time:"$((end_time-start_time))"s"
