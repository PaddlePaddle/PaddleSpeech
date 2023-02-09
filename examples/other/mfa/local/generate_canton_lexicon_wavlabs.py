import argparse
import os
import re
import shutil

import ToJyutping


def check(str):
    my_re = re.compile(r'[A-Za-z]', re.S)
    res = re.findall(my_re, str)
    if len(res):
        return True
    else:
        return False


consonants = [
    'p', 'b', 't', 'd', 'ts', 'dz', 'k', 'g', 'kw', 'gw', 'f', 'h', 'l', 'm',
    'ng', 'n', 's', 'y', 'w', 'c', 'z', 'j'
]


def get_lines(canton):
    for consonant in consonants:
        if canton.startswith(consonant):
            c, v = canton[:len(consonant)], canton[len(consonant):]
            return canton + ' ' + c + ' ' + v
    return canton + ' ' + canton


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate lexicon for Cantonese pinyin to phoneme for MFA")
    parser.add_argument(
        "--output_lexicon", type=str, help="Path to save lexicon.")
    parser.add_argument(
        "--output_wavlabs",
        type=str,
        help="Path of wavs and labs for MFA training.")
    parser.add_argument(
        "--inputs", type=str, nargs="+", help="Path to the cantonese datasets.")
    args = parser.parse_args()

    os.mkdir(args.output_wavlabs)

    utterance_info = []
    all_canton = []
    for input_ in args.inputs:
        utt = "UTTRANSINFO.txt" if "Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence" in input_ else "UTTERANCEINFO.txt"
        input_utttxt = os.path.join(input_, utt)

        with open(input_utttxt, 'r') as f:
            utterance_info = f.readlines()[1:]

        for utterance_line in utterance_info:
            _, wav_name, spk, _, text = utterance_line.split('\t')
            text = text.strip().replace(' ', '')
            # check the characters and drop the short text.
            if not check(text) and len(text) > 2:
                source_path = os.path.join(input_, 'WAV', spk, wav_name)
                target_path = os.path.join(args.output_wavlabs, wav_name)
                shutil.copy(source_path, target_path)

                lab_name = wav_name.split('.')[0] + '.lab'
                lab_target_path = os.path.join(args.output_wavlabs, lab_name)
                canton_list = ToJyutping.get_jyutping_text(text)
                with open(lab_target_path, 'w') as f:
                    f.write(canton_list)

                canton_list = canton_list.split(' ')
                all_canton.extend(canton_list)
    all_canton = set(all_canton)

    with open(args.output_lexicon, 'w') as f:
        for canton in all_canton:
            f.write(get_lines(canton) + '\n')
