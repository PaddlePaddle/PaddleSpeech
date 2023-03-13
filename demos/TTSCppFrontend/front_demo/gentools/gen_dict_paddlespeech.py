# !/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2021    liangyunming(liangyunming@baidu.com)
#
# Execute the script when PaddleSpeech has been installed
# PaddleSpeech: https://github.com/PaddlePaddle/PaddleSpeech

########################################################################

import argparse
import configparser
from paddlespeech.t2s.frontend.zh_frontend import Frontend

def get_phone(frontend, word, merge_sentences=True, print_info=False, robot=False, get_tone_ids=False):
    phonemes = frontend.get_phonemes(word, merge_sentences, print_info, robot)
    # Some optimizations
    phones, tones = frontend._get_phone_tone(phonemes[0], get_tone_ids)
    #print(type(phones), phones)
    #print(type(tones), tones)
    return phones, tones


def gen_word2phone_dict(frontend, jieba_words_dict, word2phone_dict, get_tone=False):
    with open(jieba_words_dict, "r") as f1, open(word2phone_dict, "w+") as f2:
        for line in f1.readlines():
            word = line.split(" ")[0]
            phone, tone = get_phone(frontend, word, get_tone_ids=get_tone)
            phone_str = ""

            if tone:
                assert(len(phone) == len(tone))
                for i in range(len(tone)):
                    phone_tone = phone[i] + tone[i] 
                    phone_str += (" " + phone_tone)
                phone_str = phone_str.strip("sp0").strip(" ")
            else:
                for x in phone:
                    phone_str += (" " + x)
                phone_str = phone_str.strip("sp").strip(" ")
            print(phone_str)
            f2.write(word + " " + phone_str + "\n")
    print("Generate word2phone dict successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dictionary")
    parser.add_argument(
        "--config", type=str, default="./config.ini", help="config file.")
    parser.add_argument(
        "--am_type", type=str, default="fastspeech2", help="fastspeech2 or speedyspeech")
    args = parser.parse_args()

    # Read config
    cf = configparser.ConfigParser()
    cf.read(args.config)
    jieba_words_dict_file = cf.get("jieba", "jieba_words_dict")  # get words dict

    am_type = args.am_type
    if(am_type == "fastspeech2"):
        phone2id_dict_file = cf.get(am_type, "phone2id_dict")
        word2phone_dict_file = cf.get(am_type, "word2phone_dict")

        frontend = Frontend(phone_vocab_path=phone2id_dict_file)
        print("frontend done!")

        gen_word2phone_dict(frontend, jieba_words_dict_file, word2phone_dict_file, get_tone=False)
        
    elif(am_type == "speedyspeech"):
        phone2id_dict_file = cf.get(am_type, "phone2id_dict")
        tone2id_dict_file = cf.get(am_type, "tone2id_dict")
        word2phone_dict_file = cf.get(am_type, "word2phone_dict")

        frontend = Frontend(phone_vocab_path=phone2id_dict_file, tone_vocab_path=tone2id_dict_file)
        print("frontend done!")

        gen_word2phone_dict(frontend, jieba_words_dict_file, word2phone_dict_file, get_tone=True)
        

    else:
        print("Please set correct am type, fastspeech2 or speedyspeech.")
     
    
if __name__ == "__main__":
    main()
