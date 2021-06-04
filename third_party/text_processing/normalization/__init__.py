from .sentence_split import split
from .num import RE_NUMBER, RE_FRAC, RE_PERCENTAGE, RE_RANGE, RE_INTEGER, RE_DEFAULT_NUM
from .num import replace_number, replace_frac, replace_percentage, replace_range, replace_default_num

from .chronology import RE_TIME, RE_DATE, RE_DATE2
from .chronology import replace_time, replace_date, replace_date2

from .quantifier import RE_TEMPERATURE
from .quantifier import replace_temperature

from .phone import RE_MOBILE_PHONE, RE_TELEPHONE, replace_phone

from .char_convert import tranditional_to_simplified
from .constants import F2H_ASCII_LETTERS, F2H_DIGITS, F2H_SPACE


def normalize_sentence(sentence):
    # basic character conversions
    sentence = tranditional_to_simplified(sentence)
    sentence = sentence.translate(F2H_ASCII_LETTERS).translate(
        F2H_DIGITS).translate(F2H_SPACE)

    # number related NSW verbalization
    sentence = RE_DATE.sub(replace_date, sentence)
    sentence = RE_DATE2.sub(replace_date2, sentence)
    sentence = RE_TIME.sub(replace_time, sentence)
    sentence = RE_TEMPERATURE.sub(replace_temperature, sentence)
    sentence = RE_RANGE.sub(replace_range, sentence)
    sentence = RE_FRAC.sub(replace_frac, sentence)
    sentence = RE_PERCENTAGE.sub(replace_percentage, sentence)
    sentence = RE_MOBILE_PHONE.sub(replace_phone, sentence)
    sentence = RE_TELEPHONE.sub(replace_phone, sentence)
    sentence = RE_DEFAULT_NUM.sub(replace_default_num, sentence)
    sentence = RE_NUMBER.sub(replace_number, sentence)

    return sentence


def normalize(text):
    sentences = split(text)
    sentences = [normalize_sentence(sent) for sent in sentences]
    return sentences
