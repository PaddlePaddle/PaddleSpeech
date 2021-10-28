#!/usr/bin/env bash

# Begin configuration section.
nj=4
cmd=run.pl
verbose=0
filetype=""
preprocess_conf=""
# End configuration section.

help_message=$(cat << EOF
Usage: $0 [options] <input-scp> <output-scp> [<log-dir>]
e.g.: $0 data/train/feats.scp data/train/shape.scp data/train/log
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
  --preprocess-conf <json>                         # Apply preprocess to feats when creating shape.scp
  --verbose <num>                                  # Default: 0
EOF
)

echo "$0 $*" 1>&2 # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

scp=$1
outscp=$2
data=$(dirname ${scp})
if [ $# -eq 3 ]; then
  logdir=$3
else
  logdir=${data}/log
fi
mkdir -p ${logdir}

nj=$((nj<$(<"${scp}" wc -l)?nj:$(<"${scp}" wc -l)))
split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/feats.${n}.scp"
done

utils/split_scp.pl ${scp} ${split_scps}

if [ -n "${preprocess_conf}" ]; then
    preprocess_opt="--preprocess-conf ${preprocess_conf}"
else
    preprocess_opt=""
fi
if [ -n "${filetype}" ]; then
    filetype_opt="--filetype ${filetype}"
else
    filetype_opt=""
fi

${cmd} JOB=1:${nj} ${logdir}/feat_to_shape.JOB.log \
    feat-to-shape.py --verbose ${verbose} ${preprocess_opt} ${filetype_opt} \
    scp:${logdir}/feats.JOB.scp ${logdir}/shape.JOB.scp

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${logdir}/shape.${n}.scp
done > ${outscp}

rm -f ${logdir}/feats.*.scp 2>/dev/null
