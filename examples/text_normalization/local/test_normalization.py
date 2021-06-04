import argparse
from text_processing import normalization

parser = argparse.ArgumentParser(description="Normalize text in Chinese with some rules.")
parser.add_argument("input", type=str, help="the input sentences")
parser.add_argument("output", type=str, help="path to save the output file.")
args = parser.parse_args()

with open(args.input, 'rt') as fin:
    with open(args.output, 'wt') as fout:
        for sent in fin:
            sent = normalization.normalize_sentence(sent.strip())
            fout.write(sent)
            fout.write('\n')
