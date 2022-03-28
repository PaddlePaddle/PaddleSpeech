#!/usr/bin/env python

import sys


def il(n):
    return n + 1


def ol(n):
    return n + 1


def s(n):
    return n


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    phone_count = 0
    disambig_count = 0
    for line in lines:
        sp = line.split()
        phone = sp[0]
        if phone == '<eps>' or phone == '<blank>':
            continue
        if phone.startswith('#'):
            disambig_count += 1
        else:
            phone_count += 1

    # 1. add start state
    print('0 0 {} 0'.format(il(0)))

    # 2. 0 -> i, i -> i, i -> 0
    for i in range(1, phone_count + 1):
        print('0 {} {} {}'.format(s(i), il(i), ol(i)))
        print('{} {} {} 0'.format(s(i), s(i), il(i)))
        print('{} 0 {} 0'.format(s(i), il(0)))

    # 3. i -> other phone
    for i in range(1, phone_count + 1):
        for j in range(1, phone_count + 1):
            if i != j:
                print('{} {} {} {}'.format(s(i), s(j), il(j), ol(j)))

    # 4. add disambiguous arcs on every final state
    for i in range(0, phone_count + 1):
        for j in range(phone_count + 2, phone_count + disambig_count + 2):
            print('{} {} {} {}'.format(s(i), s(i), 0, j))

    # 5. every i is final state
    for i in range(0, phone_count + 1):
        print(s(i))
