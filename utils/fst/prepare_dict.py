#!/usr/bin/env python3
import argparse


def main(args):
    # load `unit` or `vocab` file
    unit_table = set()
    with open(args.unit_file, 'r') as fin:
        for line in fin:
            unit = line.strip()
            unit_table.add(unit)

    def contain_oov(units):
        for unit in units:
            if unit not in unit_table:
                return True
        return False

    # load spm model
    bpemode = args.bpemodel
    if bpemode:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(sys.bpemodel)

    # used to filter polyphone
    lexicon_table = set()
    with open(args.in_lexicon, 'r') as fin, \
            open(args.out_lexicon, 'w') as fout:
        for line in fin:
            word = line.split()[0]
            if word == 'SIL' and not bpemode:  # `sil` might be a valid piece in bpemodel
                continue
            elif word == '<SPOKEN_NOISE>':
                continue
            else:
                # each word only has one pronunciation for e2e system
                if word in lexicon_table:
                    continue

                if bpemode:
                    pieces = sp.EncodeAsPieces(word)
                    if contain_oov(pieces):
                        print('Ignoring words {}, which contains oov unit'.
                              format(''.join(word).strip('▁')))
                        continue

                    chars = ' '.join(
                        [p if p in unit_table else '<unk>' for p in pieces])
                else:
                    # ignore words with OOV
                    if contain_oov(word):
                        print('Ignoring words {}, which contains oov unit'.
                              format(word))
                        continue

                    # Optional, append ▁ in front of english word
                    # we assume the model unit of our e2e system is char now.
                    if word.encode('utf8').isalpha() and '▁' in unit_table:
                        word = '▁' + word
                    chars = ' '.join(word)  # word is a char list

                fout.write('{} {}\n'.format(word, chars))
                lexicon_table.add(word)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FST: preprae e2e(char/spm) dict')
    parser.add_argument(
        '--unit_file',
        required=True,
        help='e2e model unit file(lang_char.txt/vocab.txt). line: char/spm_pices'
    )
    parser.add_argument(
        '--in_lexicon',
        required=True,
        help='raw lexicon file. line: word ph0 ... phn')
    parser.add_argument(
        '--out_lexicon',
        required=True,
        help='output lexicon file. line: word char0 ... charn')
    parser.add_argument('--bpemodel', default=None, help='bpemodel')

    args = parser.parse_args()
    print(args)

    main(args)
