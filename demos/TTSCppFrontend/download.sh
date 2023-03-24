#!/bin/bash
set -e

cd "$(dirname "$(realpath "$0")")"

download() {
    file="$1"
    url="$2"
    md5="$3"
    dir="$4"

    cd "$dir"

    if [ -f "$file" ] && [ "$(md5sum "$file" | awk '{ print $1 }')" = "$md5" ]; then
        echo "File $file (MD5: $md5) has been downloaded."
    else
        echo "Downloading $file..."
        wget -O "$file" "$url"

        # MD5 verify
        fileMd5="$(md5sum "$file" | awk '{ print $1 }')"
        if [ "$fileMd5" == "$md5" ]; then
            echo "File $file (MD5: $md5) has been downloaded."
        else
            echo "MD5 mismatch, file may be corrupt"
            echo "$file MD5: $fileMd5, it should be $md5"
        fi
    fi

    echo "Extracting $file..."
    echo '-----------------------'
    tar -vxf "$file"
    echo '======================='
}

########################################

DIST_DIR="$PWD/front_demo/dict"

mkdir -p "$DIST_DIR"

download 'fastspeech2_nosil_baker_ckpt_0.4.tar.gz' \
    'https://paddlespeech.bj.bcebos.com/t2s/text_frontend/fastspeech2_nosil_baker_ckpt_0.4.tar.gz' \
    '7bf1bab1737375fa123c413eb429c573' \
    "$DIST_DIR"

download 'speedyspeech_nosil_baker_ckpt_0.5.tar.gz' \
    'https://paddlespeech.bj.bcebos.com/t2s/text_frontend/speedyspeech_nosil_baker_ckpt_0.5.tar.gz' \
    '0b7754b21f324789aef469c61f4d5b8f' \
    "$DIST_DIR"

download 'jieba.tar.gz' \
    'https://paddlespeech.bj.bcebos.com/t2s/text_frontend/jieba.tar.gz' \
    '6d30f426bd8c0025110a483f051315ca' \
    "$DIST_DIR"

download 'tranditional_to_simplified.tar.gz' \
    'https://paddlespeech.bj.bcebos.com/t2s/text_frontend/tranditional_to_simplified.tar.gz' \
    '258f5b59d5ebfe96d02007ca1d274a7f' \
    "$DIST_DIR"

echo "Done."
