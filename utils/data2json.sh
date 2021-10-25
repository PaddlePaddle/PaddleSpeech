#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*" >&2 # Print the command line for logging
. ./path.sh

nj=1
cmd=run.pl
nlsyms=""
lang=""
feat="" # feat.scp
oov="<unk>"
bpecode=""
allow_one_column=false
verbose=0
trans_type=char
filetype=""
preprocess_conf=""
category=""
out="" # If omitted, write in stdout

text=""
multilingual=false

help_message=$(cat << EOF
Usage: $0 <data-dir> <dict>
e.g. $0 data/train data/lang_1char/train_units.txt
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --feat <feat-scp>                                # feat.scp or feat1.scp,feat2.scp,...
  --oov <oov-word>                                 # Default: <unk>
  --out <outputfile>                               # If omitted, write in stdout
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
  --preprocess-conf <json>                         # Apply preprocess to feats when creating shape.scp
  --verbose <num>                                  # Default: 0
EOF
)
. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

dir=$1
dic=$2
tmpdir=$(mktemp -d ${dir}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

if [ -z ${text} ]; then
    text=${dir}/text
fi

# 1. Create scp files for inputs
#   These are not necessary for decoding mode, and make it as an option
input=
if [ -n "${feat}" ]; then
    _feat_scps=$(echo "${feat}" | tr ',' ' ' )
    read -r -a feat_scps <<< $_feat_scps
    num_feats=${#feat_scps[@]}

    for (( i=1; i<=num_feats; i++ )); do
        feat=${feat_scps[$((i-1))]}
        mkdir -p ${tmpdir}/input_${i}
        input+="input_${i} "
        cat ${feat} > ${tmpdir}/input_${i}/feat.scp

        # Dump in the "legacy" style JSON format
        if [ -n "${filetype}" ]; then
            awk -v filetype=${filetype} '{print $1 " " filetype}' ${feat} \
                > ${tmpdir}/input_${i}/filetype.scp
        fi

        feat_to_shape.sh --cmd "${cmd}" --nj ${nj} \
            --filetype "${filetype}" \
            --preprocess-conf "${preprocess_conf}" \
            --verbose ${verbose} ${feat} ${tmpdir}/input_${i}/shape.scp
    done
fi

# 2. Create scp files for outputs
mkdir -p ${tmpdir}/output
if [ -n "${bpecode}" ]; then
    if [ ${multilingual} = true ]; then
        # remove a space before the language ID
        paste -d " " <(awk '{print $1}' ${text}) <(cut -f 2- -d" " ${text} \
            | spm_encode --model=${bpecode} --output_format=piece | cut -f 2- -d" ") \
            > ${tmpdir}/output/token.scp
    else
        paste -d " " <(awk '{print $1}' ${text}) <(cut -f 2- -d" " ${text} \
            | spm_encode --model=${bpecode} --output_format=piece) \
            > ${tmpdir}/output/token.scp
    fi
elif [ -n "${nlsyms}" ]; then
    text2token.py -s 1 -n 1 -l ${nlsyms} ${text} --trans_type ${trans_type} > ${tmpdir}/output/token.scp
else
    text2token.py -s 1 -n 1 ${text} --trans_type ${trans_type} > ${tmpdir}/output/token.scp
fi
< ${tmpdir}/output/token.scp utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/output/tokenid.scp
# +2 comes from CTC blank and EOS
vocsize=$(tail -n 1 ${dic} | awk '{print $2}')
odim=$(echo "$vocsize + 2" | bc)
< ${tmpdir}/output/tokenid.scp awk -v odim=${odim} '{print $1 " " NF-1 "," odim}' > ${tmpdir}/output/shape.scp

cat ${text} > ${tmpdir}/output/text.scp


# 3. Create scp files for the others
mkdir -p ${tmpdir}/other
if [ ${multilingual} == true ]; then
    awk '{
        n = split($1,S,"[-]");
        lang=S[n];
        print $1 " " lang
    }' ${text} > ${tmpdir}/other/lang.scp
elif [ -n "${lang}" ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${text} > ${tmpdir}/other/lang.scp
fi

if [ -n "${category}" ]; then
    awk -v category=${category} '{print $1 " " category}' ${dir}/text \
        > ${tmpdir}/other/category.scp
fi
cat ${dir}/utt2spk > ${tmpdir}/other/utt2spk.scp

# 4. Merge scp files into a JSON file
opts=""
if [ -n "${feat}" ]; then
    intypes="${input} output other"
else
    intypes="output other"
fi
for intype in ${intypes}; do
    if [ -z "$(find "${tmpdir}/${intype}" -name "*.scp")" ]; then
        continue
    fi

    if [ ${intype} != other ]; then
        opts+="--${intype%_*}-scps "
    else
        opts+="--scps "
    fi

    for x in "${tmpdir}/${intype}"/*.scp; do
        k=$(basename ${x} .scp)
        if [ ${k} = shape ]; then
            opts+="shape:${x}:shape "
        else
            opts+="${k}:${x} "
        fi
    done
done

if ${allow_one_column}; then
    opts+="--allow-one-column true "
else
    opts+="--allow-one-column false "
fi

if [ -n "${out}" ]; then
    opts+="-O ${out}"
fi
merge_scp2json.py --verbose ${verbose} ${opts}

rm -fr ${tmpdir}
