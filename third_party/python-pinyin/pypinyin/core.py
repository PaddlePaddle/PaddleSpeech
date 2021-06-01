from itertools import chain
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Text
from typing import Union

from pypinyin.constants import PHRASES_DICT
from pypinyin.constants import PINYIN_DICT
from pypinyin.constants import Style
from pypinyin.converter import _mixConverter
from pypinyin.converter import Converter
from pypinyin.converter import DefaultConverter
from pypinyin.seg import mmseg
from pypinyin.seg import simpleseg
from pypinyin.utils import (_replace_tone2_style_dict_to_default)
import jieba

TStyle = Style
TErrors = Union[Callable[[Text], Text], Text]
TPinyinResult = List[List[Text]]


def load_single_dict(pinyin_dict: Dict[int, Text], style: str='default'):
    """载入用户自定义的单字拼音库
    :param pinyin_dict: 单字拼音库。比如： ``{0x963F: u"ā,ē"}``
    :param style: pinyin_dict 参数值的拼音库风格. 支持 'default', 'tone2'
    :type pinyin_dict: dict
    """
    if style == 'tone2':
        for k, v in pinyin_dict.items():
            v = _replace_tone2_style_dict_to_default(v)
            PINYIN_DICT[k] = v
    else:
        PINYIN_DICT.update(pinyin_dict)

    mmseg.retrain(mmseg.seg)


def load_phrases_dict(phrases_dict: Dict[Text, List[List[Text]]],
                      style: str='default'):
    """载入用户自定义的词语拼音库
    :param phrases_dict: 词语拼音库。比如： ``{u"阿爸": [[u"ā"], [u"bà"]]}``
    :param style: phrases_dict 参数值的拼音库风格. 支持 'default', 'tone2'
    :type phrases_dict: dict
    """
    if style == 'tone2':
        for k, value in phrases_dict.items():
            v = [
                list(map(_replace_tone2_style_dict_to_default, pys))
                for pys in value
            ]
            PHRASES_DICT[k] = v
    else:
        PHRASES_DICT.update(phrases_dict)

    mmseg.retrain(mmseg.seg)


class Pinyin():
    def __init__(self, converter: Converter=None, **kwargs: Any):
        self._converter = converter or DefaultConverter()

    def pinyin(self,
               hans: Union[List[Text], Text],
               style: TStyle=Style.TONE,
               heteronym: bool=False,
               errors: TErrors='default',
               strict: bool=True,
               **kwargs: Any) -> TPinyinResult:
        """将汉字转换为拼音，返回汉字的拼音列表。
        :param hans: 汉字字符串( ``'你好吗'`` )或列表( ``['你好', '吗']`` ).
                     可以使用自己喜爱的分词模块对字符串进行分词处理,
                     只需将经过分词处理的字符串列表传进来就可以了。
        :type hans: unicode 字符串或字符串列表
        :param style: 指定拼音风格，默认是 :py:attr:`~pypinyin.Style.TONE` 风格。
                      更多拼音风格详见 :class:`~pypinyin.Style`
        :param errors: 指定如何处理没有拼音的字符。详见 :ref:`handle_no_pinyin`
                       * ``'default'``: 保留原始字符
                       * ``'ignore'``: 忽略该字符
                       * ``'replace'``: 替换为去掉 ``\\u`` 的 unicode 编码字符串
                         (``'\\u90aa'`` => ``'90aa'``)
                       * callable 对象: 回调函数之类的可调用对象。
        :param heteronym: 是否启用多音字
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :return: 拼音列表
        :rtype: list
        """
        # 对字符串进行分词处理
        if isinstance(hans, str):
            han_list = self.seg(hans)
        else:
            han_list = chain(*(self.seg(x) for x in hans))

        pys = []
        for words in han_list:
            pys.extend(
                self._converter.convert(
                    words, style, heteronym, errors, strict=strict))
        return pys

    def lazy_pinyin(self,
                    hans: Union[List[Text], Text],
                    style: TStyle=Style.NORMAL,
                    errors: TErrors='default',
                    strict: bool=True,
                    **kwargs: Any) -> List[Text]:
        """将汉字转换为拼音，返回不包含多音字结果的拼音列表.
        与 :py:func:`~pypinyin.pinyin` 的区别是每个汉字的拼音是个字符串，
        并且每个字只包含一个读音.
        :param hans: 汉字
        :type hans: unicode or list
        :param style: 指定拼音风格，默认是 :py:attr:`~pypinyin.Style.NORMAL` 风格。
                      更多拼音风格详见 :class:`~pypinyin.Style`。
        :param errors: 指定如何处理没有拼音的字符，详情请参考
                       :py:func:`~pypinyin.pinyin`
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :return: 拼音列表(e.g. ``['zhong', 'guo', 'ren']``)
        :rtype: list
        """
        return list(
            chain(*self.pinyin(
                hans,
                style=style,
                heteronym=False,
                errors=errors,
                strict=strict)))

    def pre_seg(self, hans: Text, **kwargs: Any) -> Optional[List[Text]]:
        """对字符串进行分词前将调用 ``pre_seg`` 方法对未分词的字符串做预处理。
        默认原样返回传入的 ``hans``。
        如果这个方法的返回值类型是 ``list``，表示返回的是一个分词后的结果，此时，
        ``seg`` 方法中将不再调用 ``seg_function`` 进行分词。
        :param hans: 分词前的字符串
        :return: ``None`` or ``list``
        """
        outs = jieba.lcut(hans)  # 默认用jieba分词，从语义角度分词。
        return outs

    def post_seg(self, hans: Text, seg_data: List[Text],
                 **kwargs: Any) -> Optional[List[Text]]:
        """对字符串进行分词后将调用 ``post_seg`` 方法对分词后的结果做处理。
        默认原样返回传入的 ``seg_data``。
        如果这个方法的返回值类型是 ``list``，表示对分词结果做了二次处理，此时，
        ``seg`` 方法将以这个返回的数据作为返回值。
        :param hans: 分词前的字符串
        :param seg_data: 分词后的结果
        :type seg_data: list
        :return: ``None`` or ``list``
        """
        pass

    def seg(self, hans: Text, **kwargs: Any) -> List[Text]:
        """对汉字进行分词。
        分词前会调用 ``pre_seg`` 方法，分词后会调用 ``post_seg`` 方法。
        :param hans:
        :return:
        """
        pre_data = self.pre_seg(hans)
        if isinstance(pre_data, list):
            seg_data = pre_data
        else:
            seg_data = self.get_seg()(hans)

        post_data = self.post_seg(hans, seg_data)
        if isinstance(post_data, list):
            return post_data

        return seg_data

    def get_seg(self, **kwargs: Any) -> Callable[[Text], List[Text]]:
        """获取分词函数。
        :return: 分词函数
        """
        return simpleseg.seg


_default_convert = DefaultConverter()
_default_pinyin = Pinyin(_default_convert)


def pinyin(hans: Union[List[Text], Text],
           style: TStyle=Style.TONE,
           heteronym: bool=False,
           errors: TErrors='default',
           strict: bool=True,
           v_to_u: bool=False,
           neutral_tone_with_five=False) -> List[List[Text]]:
    """将汉字转换为拼音，返回汉字的拼音列表。
    :param hans: 汉字字符串( ``'你好吗'`` )或列表( ``['你好', '吗']`` ).
                 可以使用自己喜爱的分词模块对字符串进行分词处理,
                 只需将经过分词处理的字符串列表传进来就可以了。
    :type hans: unicode 字符串或字符串列表
    :param style: 指定拼音风格，默认是 :py:attr:`~pypinyin.Style.TONE` 风格。
                  更多拼音风格详见 :class:`~pypinyin.Style`
    :param errors: 指定如何处理没有拼音的字符。详见 :ref:`handle_no_pinyin`
                   * ``'default'``: 保留原始字符
                   * ``'ignore'``: 忽略该字符
                   * ``'replace'``: 替换为去掉 ``\\u`` 的 unicode 编码字符串
                     (``'\\u90aa'`` => ``'90aa'``)
                   * callable 对象: 回调函数之类的可调用对象。
    :param heteronym: 是否启用多音字
    :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                   是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :param v_to_u: 无声调相关拼音风格下的结果是否使用 ``ü`` 代替原来的 ``v``
    :type v_to_u: bool
    :param neutral_tone_with_five: 声调使用数字表示的相关拼音风格下的结果是否
                                   使用 5 标识轻声
    :type neutral_tone_with_five: bool
    :return: 拼音列表
    :rtype: list
    :raise AssertionError: 当传入的字符串不是 unicode 字符时会抛出这个异常
    Usage::
      >>> from pypinyin import pinyin, Style
      >>> import pypinyin
      >>> pinyin('中心')
      [['zhōng'], ['xīn']]
      >>> pinyin('中心', heteronym=True)  # 启用多音字模式
      [['zhōng', 'zhòng'], ['xīn']]
      >>> pinyin('中心', style=Style.FIRST_LETTER)  # 设置拼音风格
      [['z'], ['x']]
      >>> pinyin('中心', style=Style.TONE2)
      [['zho1ng'], ['xi1n']]
      >>> pinyin('中心', style=Style.CYRILLIC)
      [['чжун1'], ['синь1']]
      >>> pinyin('战略', v_to_u=True, style=Style.NORMAL)
      [['zhan'], ['lüe']]
      >>> pinyin('衣裳', style=Style.TONE3, neutral_tone_with_five=True)
      [['yi1'], ['shang5']]
    """
    _pinyin = Pinyin(
        _mixConverter(
            v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five))

    return _pinyin.pinyin(
        hans, style=style, heteronym=heteronym, errors=errors, strict=strict)


def slug(hans: Union[List[Text], Text],
         style: TStyle=Style.NORMAL,
         heteronym: bool=False,
         separator: Text='-',
         errors: TErrors='default',
         strict: bool=True) -> Text:
    """将汉字转换为拼音，然后生成 slug 字符串.
    :param hans: 汉字
    :type hans: unicode or list
    :param style: 指定拼音风格，默认是 :py:attr:`~pypinyin.Style.NORMAL` 风格。
                  更多拼音风格详见 :class:`~pypinyin.Style`
    :param heteronym: 是否启用多音字
    :param separator: 两个拼音间的分隔符/连接符
    :param errors: 指定如何处理没有拼音的字符，详情请参考
                   :py:func:`~pypinyin.pinyin`
    :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                   是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :return: slug 字符串.
    :raise AssertionError: 当传入的字符串不是 unicode 字符时会抛出这个异常
    ::
      >>> import pypinyin
      >>> from pypinyin import Style
      >>> pypinyin.slug('中国人')
      'zhong-guo-ren'
      >>> pypinyin.slug('中国人', separator=' ')
      'zhong guo ren'
      >>> pypinyin.slug('中国人', style=Style.FIRST_LETTER)
      'z-g-r'
      >>> pypinyin.slug('中国人', style=Style.CYRILLIC)
      'чжун1-го2-жэнь2'
    """
    return separator.join(
        chain(*_default_pinyin.pinyin(
            hans,
            style=style,
            heteronym=heteronym,
            errors=errors,
            strict=strict)))


def lazy_pinyin(hans: Union[List[Text], Text],
                style: TStyle=Style.NORMAL,
                errors: TErrors='default',
                strict: bool=True,
                v_to_u: bool=False,
                neutral_tone_with_five: bool=False) -> List[Text]:
    """将汉字转换为拼音，返回不包含多音字结果的拼音列表.
    与 :py:func:`~pypinyin.pinyin` 的区别是返回的拼音是个字符串，
    并且每个字只包含一个读音.
    :param hans: 汉字
    :type hans: unicode or list
    :param style: 指定拼音风格，默认是 :py:attr:`~pypinyin.Style.NORMAL` 风格。
                  更多拼音风格详见 :class:`~pypinyin.Style`。
    :param errors: 指定如何处理没有拼音的字符，详情请参考
                   :py:func:`~pypinyin.pinyin`
    :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                   是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :param v_to_u: 无声调相关拼音风格下的结果是否使用 ``ü`` 代替原来的 ``v``
    :type v_to_u: bool
    :param neutral_tone_with_five: 声调使用数字表示的相关拼音风格下的结果是否
                                   使用 5 标识轻声
    :type neutral_tone_with_five: bool
    :return: 拼音列表(e.g. ``['zhong', 'guo', 'ren']``)
    :rtype: list
    :raise AssertionError: 当传入的字符串不是 unicode 字符时会抛出这个异常
    Usage::
      >>> from pypinyin import lazy_pinyin, Style
      >>> import pypinyin
      >>> lazy_pinyin('中心')
      ['zhong', 'xin']
      >>> lazy_pinyin('中心', style=Style.TONE)
      ['zhōng', 'xīn']
      >>> lazy_pinyin('中心', style=Style.FIRST_LETTER)
      ['z', 'x']
      >>> lazy_pinyin('中心', style=Style.TONE2)
      ['zho1ng', 'xi1n']
      >>> lazy_pinyin('中心', style=Style.CYRILLIC)
      ['чжун1', 'синь1']
      >>> lazy_pinyin('战略', v_to_u=True)
      ['zhan', 'lüe']
      >>> lazy_pinyin('衣裳', style=Style.TONE3, neutral_tone_with_five=True)
      ['yi1', 'shang5']
    """
    _pinyin = Pinyin(
        _mixConverter(
            v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five))

    return _pinyin.lazy_pinyin(hans, style=style, errors=errors, strict=strict)
