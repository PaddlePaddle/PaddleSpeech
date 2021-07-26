#!/usr/bin/env python3
import argparse


def il(n):
    """ilabel"""
    return n + 1


def ol(n):
    """olabel"""
    return n + 1


def s(n):
    """state"""
    return n


def main(args):
    with open(args.token_file) as f:
        lines = f.readlines()
    # token count w/0 <blank> <eps>
    phone_count = 0
    disambig_count = 0
    for line in lines:
        sp = line.strip().split()
        phone = sp[0]
        if phone == '<eps>' or phone == '<blank>':
            continue
        if phone.startswith('#'):
            disambig_count += 1
        else:
            phone_count += 1

    # 1. add start state
    # first token is <blank>:0
    print('0 0 {} 0'.format(il(0)))

    # 2. 0 -> i, i -> i, i -> 0
    # non-blank token start from 1
    for i in range(1, phone_count + 1):
        # eating `token`
        print('0 {} {} {}'.format(s(i), il(i), ol(i)))
        # remove repeating `token`
        print('{} {} {} 0'.format(s(i), s(i), il(i)))
        # skip ending <blank> `token`
        print('{} 0 {} 0'.format(s(i), il(0)))

    # 3. i -> other phone
    # non-blank token to other non-blank token
    for i in range(1, phone_count + 1):
        for j in range(1, phone_count + 1):
            if i != j:
                print('{} {} {} {}'.format(s(i), s(j), il(j), ol(j)))

    # 4. add disambiguous arcs on every final state
    # blank and non-blank token maybe ending with disambiguous `token`
    for i in range(0, phone_count + 1):
        for j in range(phone_count + 2, phone_count + disambig_count + 2):
            print('{} {} {} {}'.format(s(i), s(i), 0, j))

    # 5. every i is final state
    # blank and non-blank `token` are final state
    for i in range(0, phone_count + 1):
        print(s(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='FST: CTC Token unfold FST transducer')
    parser.add_argument(
        '--token_file',
        required=True,
        help='e2e model token file. line: token(char/phone/spm/disambigous)')
    args = parser.parse_args()

    main(args)
