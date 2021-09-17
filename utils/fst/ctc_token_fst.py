#!/usr/bin/env python3
import argparse


def main(args):
    """Token Transducer"""
    # <eps> entry
    print('0 1 <eps> <eps>')
    # skip begining and ending <blank>
    print('1 1 <blank> <eps>')
    print('2 2 <blank> <eps>')
    # <eps> exit
    print('2 0 <eps> <eps>')

    # linking `token` between node 1 and node 2
    with open(args.token_file, 'r') as fin:
        node = 3
        for entry in fin:
            fields = entry.strip().split(' ')
            phone = fields[0]
            if phone == '<eps>' or phone == '<blank>':
                continue
            elif '#' in phone:
                # disambiguous phone
                # `token` maybe ending with disambiguous symbol
                print('{} {} {} {}'.format(0, 0, '<eps>', phone))
            else:
                # eating `token`
                print('{} {} {} {}'.format(1, node, phone, phone))
                # remove repeating `token`
                print('{} {} {} {}'.format(node, node, phone, '<eps>'))
                # leaving `token`
                print('{} {} {} {}'.format(node, 2, '<eps>', '<eps>'))
            node += 1
    # Fianl node
    print('0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FST: CTC Token FST transducer')
    parser.add_argument(
        '--token_file',
        required=True,
        help='e2e model token file. line: token(char/phone/spm/disambigous)')

    args = parser.parse_args()

    main(args)
