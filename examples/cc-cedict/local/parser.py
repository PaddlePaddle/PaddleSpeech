# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://github.com/rubber-duck-dragon/rubber-duck-dragon.github.io/blob/master/cc-cedict_parser/parser.py
#A parser for the CC-Cedict. Convert the Chinese-English dictionary into a list of python dictionaries with "traditional","simplified", "pinyin", and "english" keys.
#Make sure that the cedict_ts.u8 file is in the same folder as this file, and that the name matches the file name on line 13.
#Before starting, open the CEDICT text file and delete the copyright information at the top. Otherwise the program will try to parse it and you will get an error message.
#Characters that are commonly used as surnames have two entries in CC-CEDICT. This program will remove the surname entry if there is another entry for the character. If you want to include the surnames, simply delete lines 59 and 60.
#This code was written by Franki Allegra in February 2020.
import json
import sys

# usage: bin ccedict dump.json

with open(sys.argv[1], 'rt') as file:
    text = file.read()
    lines = text.split('\n')
    dict_lines = list(lines)

    def parse_line(line):
        parsed = {}
        if line == '':
            dict_lines.remove(line)
            return 0
        if line.startswith('#'):
            return 0
        if line.startswith('%'):
            return 0
        line = line.rstrip('/')
        line = line.split('/')
        if len(line) <= 1:
            return 0
        english = line[1]
        char_and_pinyin = line[0].split('[')
        characters = char_and_pinyin[0]
        characters = characters.split()
        traditional = characters[0]
        simplified = characters[1]
        pinyin = char_and_pinyin[1]
        pinyin = pinyin.rstrip()
        pinyin = pinyin.rstrip("]")
        parsed['traditional'] = traditional
        parsed['simplified'] = simplified
        parsed['pinyin'] = pinyin
        parsed['english'] = english
        list_of_dicts.append(parsed)

    def remove_surnames():
        for x in range(len(list_of_dicts) - 1, -1, -1):
            if "surname " in list_of_dicts[x]['english']:
                if list_of_dicts[x]['traditional'] == list_of_dicts[x + 1][
                        'traditional']:
                    list_of_dicts.pop(x)

    def main():

        #make each line into a dictionary
        print("Parsing dictionary . . .")
        for line in dict_lines:
            parse_line(line)

        #remove entries for surnames from the data (optional):
        print("Removing Surnames . . .")
        remove_surnames()

        print("Saving to database (this may take a few minutes) . . .")
        with open(sys.argv[2], 'wt') as fout:
            for one_dict in list_of_dicts:
                json_str = json.dumps(one_dict)
                fout.write(json_str + "\n")
        print('Done!')


list_of_dicts = []
parsed_dict = main()
