"""
### chinese
适用于中文、英文和中英混合的音素，其中汉字拼音采用清华大学的音素，英文字符分字母和英文。

- 中文音素简介：

```
声母：
aa b c ch d ee f g h ii j k l m n oo p q r s sh t uu vv x z zh

韵母：
a ai an ang ao e ei en eng er i ia ian iang iao ie in ing iong iu ix iy iz o ong ou u ua uai uan uang ueng ui un uo v van ve vn ng uong

声调：
1 2 3 4 5

字母：
Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj Kk Ll Mm Nn Oo Pp Qq Rr Ss Tt Uu Vv Ww Xx Yy Zz

英文：
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

标点：
! ? . , ; : " # ( )
注：!=!！|?=?？|.=.。|,=,，、|;=;；|:=:：|"="“|#=# 　\t|(=(（[［{｛【<《|)=)）]］}｝】>》

预留：
w y 0 6 7 8 9
注：w=%|y=$|0=0|6=6|7=7|8=8|9=9

其他：
_ ~  - *
```
"""
from .convert import fan2jian, jian2fan, quan2ban, ban2quan
from .number import say_digit, say_decimal, say_number
from .pinyin import text2pinyin, split_pinyin
from .sequence import text2sequence, text2phoneme, pinyin2phoneme, phoneme2sequence, sequence2phoneme, change_diao
from .sequence import symbol_chinese, ph2id_dict, id2ph_dict

from .symbol import symbol_chinese as symbols
from .phoneme import shengyun2ph_dict


def text_to_sequence(src, cleaner_names=None,  **kwargs):
    """
    文本样例：卡尔普陪外孙玩滑梯。
    拼音样例：ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1 .
    :param src: str,拼音或文本字符串
    :param cleaner_names: 文本处理方法选择，暂时提供拼音和文本两种方法。
    :return: list,ID列表
    """
    if cleaner_names == "pinyin":
        pys = []
        for py in src.split():
            if py.isalnum():
                pys.append(py)
            else:
                pys.append((py,))
        phs = pinyin2phoneme(pys)
        phs = change_diao(phs)
        seq = phoneme2sequence(phs)
        return seq
    else:
        return text2sequence(src)


def sequence_to_text(src):
    out = sequence2phoneme(src)
    return " ".join(out)


if __name__ == "__main__":
    print(__file__)
    text = "ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1 . "
    out = text_to_sequence(text)
    print(out)
    out = sequence_to_text(out)
    print(out)
