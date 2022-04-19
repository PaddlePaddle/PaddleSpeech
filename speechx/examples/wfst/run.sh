#!/bin/bash
set -eo pipefail

. path.sh

stage=-1
stop_stage=100

. utils/parse_options.sh

if ! which fstprint ; then
    pushd $MAIN_ROOT/tools
    make kaldi.done
    popd
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
    # build T & L
    # utils/fst/compile_lexicon_token_fst.sh <dict-src-dir> <tmp-dir> <lang-dir>
    utils/fst/compile_lexicon_token_fst.sh \
        data/local/dict data/local/tmp data/local/lang

    # build G & LG & TLG
    # utils/fst/make_tlg.sh <lm_dir> <src_lang> <tgt_lang>
    utils/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;
fi

echo "build TLG done."
exit 0
