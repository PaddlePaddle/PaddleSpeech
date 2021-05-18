import argparse
import re
import jieba
import pypinyin
from pypinyin import lazy_pinyin, Style

def extract_pinyin(source, target, use_jieba=False):
    with open(source, 'rt', encoding='utf-8') as f:
        with open(target, 'wt', encoding='utf-8') as g:
            for i, line in enumerate(f):
                if i % 2 == 0:
                    g.write(line)
                    sentence_id, raw_text = line.strip().split()
                    raw_text = re.sub(r'#\d', '', raw_text)
                    if use_jieba:
                        raw_text = jieba.lcut(raw_text)
                    syllables = lazy_pinyin(raw_text, errors='ignore', style=Style.TONE3, neutral_tone_with_five=True)
                    transcription = ' '.join(syllables)
                    g.write(f'\t{transcription}\n')
                else:
                    continue
                    
                
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract baker pinyin labels")
    parser.add_argument("input", type=str, help="source file of baker's prosody label file")
    parser.add_argument("output", type=str, help="target file to write pinyin lables")
    parser.add_argument("--use-jieba", action='store_true', help="use jieba for word segmentation.")
    args = parser.parse_args()
    print(args)
    extract_pinyin(args.input, args.output, use_jieba=args.use_jieba)
