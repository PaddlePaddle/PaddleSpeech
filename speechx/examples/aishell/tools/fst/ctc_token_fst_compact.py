#!/usr/bin/env python

import sys

print('0 0 <blank> <eps>')

with open(sys.argv[1], 'r', encoding='utf8') as fin:
    node = 1
    for entry in fin:
        fields = entry.strip().split(' ')
        phone = fields[0]
        if phone == '<eps>' or phone == '<blank>':
            continue
        elif '#' in phone:  # disambiguous phone
            print('{} {} {} {}'.format(0, 0, '<eps>', phone))
        else:
            print('{} {} {} {}'.format(0, node, phone, phone))
            print('{} {} {} {}'.format(node, node, phone, '<eps>'))
            print('{} {} {} {}'.format(node, 0, '<eps>', '<eps>'))
        node += 1
print('0')
