#!/usr/bin/env python3

from typing import List, Text
import sys
import jieba
import string
import re
from zhon import hanzi

def char_token(s: Text) -> List[Text]:
    return list(s)

def word_token(s: Text) -> List[Text]:
    return jieba.lcut(s)

def tn(s: Text) -> Text:
    s = s.strip()
    s = s.replace('*', '')
    # rm english punctuations
    s = re.sub(f'[re.escape(string.punctuation)]' , "", s)
    # rm chinese punctuations
    s = re.sub(f'[{hanzi.punctuation}]', "", s)
    # text normalization
    
    # rm english
    s = ''.join(re.findall(hanzi.sent, s))
    return s

def main(infile, outfile, tokenizer=None):
    with open(infile, 'rt') as fin, open(outfile, 'wt') as fout:
        lines = fin.readlines()
        for l in lines:
            l = tn(l)
            if tokenizer:
                l = ' '.join(tokenizer(l))
            fout.write(l)
            fout.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"sys.arv[0] [char|word] text text_out ")
        exit(-1)

    token_type = sys.argv[1]
    text = sys.argv[2]
    text_out = sys.argv[3]

    if token_type == 'char':
        tokenizer = char_token
    elif token_type == 'word':
        tokenizer = word_token
    else:
        tokenizer = None

    main(text, text_out, tokenizer)