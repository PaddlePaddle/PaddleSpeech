#!/usr/bin/env python

import sys

print('0 1 <eps> <eps>')
print('1 1 <blank> <eps>')
print('2 2 <blank> <eps>')
print('2 0 <eps> <eps>')

with open(sys.argv[1], 'r') as fin:
    node = 3
    for entry in fin:
        fields = entry.strip().split(' ')
        phone = fields[0]
        if phone == '<eps>' or phone == '<blank>':
            continue
        elif '#' in phone:  # disambiguous phone
            print('{} {} {} {}'.format(0, 0, '<eps>', phone))
        else:
            print('{} {} {} {}'.format(1, node, phone, phone))
            print('{} {} {} {}'.format(node, node, phone, '<eps>'))
            print('{} {} {} {}'.format(node, 2, '<eps>', '<eps>'))
        node += 1
print('0')
