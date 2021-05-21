"""
### pinyinkit
文本转拼音的模块，依赖python-pinyin，jieba，phrase-pinyin-data模块。
"""
import re
from pypinyin import lazy_pinyin, Style, load_phrases_dict, load_phrases_dict


# def parse_pinyin_txt(inpath):
#     # U+4E2D: zhōng,zhòng  # 中
#     outs = []
#     with open(inpath, encoding="utf8") as fin:
#         for line in tqdm(fin, desc='load pinyin', ncols=80, mininterval=1):
#             if line.startswith("#"):
#                 continue
#             res = _ziyin_re.search(line)
#             if res:
#                 zi = res.group(3).strip()
#                 if len(zi) == 1:
#                     outs.append([zi, res.group(2).strip().split(",")])
#                 else:
#                     print(line)
#             elif line.strip():
#                 print(line)
#     return {ord(z): ','.join(p) for z, p in outs}


# def parse_phrase_txt(inpath):
#     # 一一对应: yī yī duì yìng
#     outs = []
#     with open(inpath, encoding="utf8") as fin:
#         for line in tqdm(fin, desc='load phrase', ncols=80, mininterval=1):
#             if line.startswith("#"):
#                 continue
#             parts = line.split(":")
#             zs = parts[0].strip()
#             ps = parts[1].strip().split()
#             if len(parts) == 2 and len(zs) == len(ps) and len(zs) >= 2:
#                 outs.append([zs, ps])
#             elif line.strip():
#                 print(line)
#     return {zs: [[p] for p in ps] for zs, ps in outs}


# def initialize():
#     # 导入数据
#     inpath = Path(__file__).absolute().parent.joinpath('phrase_pinyin.txt.py')
#     _phrases_dict = parse_phrase_txt(inpath)
#     load_phrases_dict(_phrases_dict)  # big:398815 small:36776

#     inpath = Path(__file__).absolute().parent.joinpath('single_pinyin.txt.py')
#     _pinyin_dict = parse_pinyin_txt(inpath)
#     load_single_dict(_pinyin_dict)  # 41451

#     jieba.initialize()
#     # for word, _ in tqdm(_phrases_dict.items(), desc='jieba add word', ncols=80, mininterval=1):
#     #     jieba.add_word(word)


# 兼容0.1.0之前的版本。
# 音调：5为轻声
_diao_re = re.compile(r"([12345]$)")


def text2pinyin(text, errors=None, **kwargs):
    """
    汉语文本转为拼音列表
    :param text: str,汉语文本字符串
    :param errors: function,对转拼音失败的字符的处理函数，默认保留原样
    :return: list,拼音列表
    """
    if errors is None:
        errors = default_errors
    pin = lazy_pinyin(text, style=Style.TONE3, errors=errors, strict=True, neutral_tone_with_five=True, **kwargs)
    return pin


def default_errors(x):
    return list(x)


def split_pinyin(py):
    """
    单个拼音转为音素列表
    :param py: str,拼音字符串
    :param errors: function,对OOV拼音的处理函数，默认保留原样
    :return: list,音素列表
    """
    parts = _diao_re.split(py)
    if len(parts) == 1:
        fuyuan = py
        diao = "5"
    else:
        fuyuan = parts[0]
        diao = parts[1]
    return [fuyuan, diao]


if __name__ == "__main__":
    print(__file__)
    assert text2pinyin("拼音") == ['pin1', 'yin1']
    assert text2pinyin("汉字,a1") == ['han4', 'zi4', ',', 'a', '1']
