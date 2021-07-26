#!/usr/bin/env python3
import argparse


def main(args):
    # skip <blank> `token`
    print('0 0 <blank> <eps>')

    with open(args.token_file, 'r') as fin:
        for entry in fin:
            fields = entry.strip().split(' ')
            phone = fields[0]
            if phone == '<eps>' or phone == '<blank>':
                continue
            elif '#' in phone:
                # disambiguous phone
                # maybe add disambiguous `token`
                print('{} {} {} {}'.format(0, 0, '<eps>', phone))
            else:
                # eating `token`
                print('{} {} {} {}'.format(0, 0, phone, phone))

    # final state
    print('0')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='FST: RNN-T Token FST transducer')
    parser.add_argument(
        '--token_file',
        required=True,
        help='e2e model token file. line: token(char/phone/spm/disambigous)')
    args = parser.parse_args()

    main(args)
