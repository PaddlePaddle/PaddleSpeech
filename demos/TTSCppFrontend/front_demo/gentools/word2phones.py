from pypinyin import lazy_pinyin, Style
import re

worddict = "./dict/jieba_part.dict.utf8"
newdict = "./dict/word_phones.dict"

def GenPhones(initials, finals, seperate=True):

    phones = []
    for c, v in zip(initials, finals):
        if re.match(r'i\d', v):
            if c in ['z', 'c', 's']:
                v = re.sub('i', 'ii', v)
            elif c in ['zh', 'ch', 'sh', 'r']:
                v = re.sub('i', 'iii', v)
        if c:
            if seperate == True:
                phones.append(c + '0')
            elif seperate == False:
                phones.append(c)
            else:
                print("Not sure whether phone and tone need to be separated")
        if v:
            phones.append(v)
    return phones


with open(worddict, "r") as f1, open(newdict, "w+") as f2:
    for line in f1.readlines():
        word = line.split(" ")[0]
        initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
        finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

        phones = GenPhones(initials, finals, True)

        temp = " ".join(phones)
        f2.write(word + " " + temp + "\n")
