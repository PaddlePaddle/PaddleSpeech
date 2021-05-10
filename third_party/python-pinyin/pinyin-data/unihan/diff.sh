#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

function main() {
    printf '%-14s  %-8s  %-8s\n' '' 'parsed' 'Unihan'
    for kind in 'kHanyuPinyin' 'kMandarin' 'kHanyuPinlu' 'kXHC1983'
    do
        unihanCount=$(less Unihan_Readings.txt |grep -v '^#' |grep -c "$kind")
        parsedCount=$(less "$kind".txt | grep -c "")
        printf '%-14s  %-8s  %-8s\n' "$kind" "$parsedCount" "$unihanCount"
    done
}
main
