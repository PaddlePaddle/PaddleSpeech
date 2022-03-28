#!/usr/bin/env python

import sys

print('0 0 <blank> <eps>')

with open(sys.argv[1], 'r', encoding='utf8') as fin:
    for entry in fin:
        fields = entry.strip().split(' ')
        phone = fields[0]
        if phone == '<eps>' or phone == '<blank>':
            continue
        elif '#' in phone:  # disambiguous phone
            print('{} {} {} {}'.format(0, 0, '<eps>', phone))
        else:
            print('{} {} {} {}'.format(0, 0, phone, phone))
print('0')
