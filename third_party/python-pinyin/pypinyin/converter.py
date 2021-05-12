from copy import deepcopy
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Text
from typing import Union

from pypinyin.constants import PHRASES_DICT
from pypinyin.constants import PINYIN_DICT
from pypinyin.constants import RE_HANS
from pypinyin.constants import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.contrib.uv import V2UMixin
from pypinyin.style import auto_discover
from pypinyin.style import convert as convert_style
from pypinyin.utils import _remove_dup_items

TStyle = Style
TErrors = Union[Callable[[Text], Text], Text]
TPinyinResult = List[List[Text]]
TErrorResult = Union[Text, List[Text], None]
TNoPinyinResult = Union[TPinyinResult, List[Text], Text, None]

auto_discover()


class Converter():
    def convert(self,
                words: Text,
                style: TStyle,
                heteronym: bool,
                errors: TErrors,
                strict: bool=...,
                **kwargs: Any) -> TPinyinResult:
        # TODO: use ``abc`` module
        raise NotImplementedError  # pragma: no cover


class DefaultConverter(Converter):
    def __init__(self, **kwargs: Any) -> None:
        pass

    def post_pinyin(self,
                    han: Text,
                    heteronym: bool,
                    pinyin: TPinyinResult,
                    **kwargs: Any) -> Union[TPinyinResult, None]:
        """找到汉字对应的拼音后，会调用 ``post_pinyin`` 方法。
        如果返回值不为 ``None`` 会使用返回的结果作为 han 的拼音数据。
        :param han: 单个汉字或者词语
        :param heteronym: 是否需要处理多音字
        :param pinyin: 单个汉字的拼音数据或词语的拼音数据 list
        :type pinyin: list
        :param kwargs: 其他关键字参数，暂时无用，用于以后扩展新的参数。
        :return: ``None`` 或代替 ``pinyin`` 作为 han 的拼音 list。
        """
        pass

    def _single_pinyin(self,
                       han: Text,
                       style: TStyle,
                       heteronym: bool,
                       errors: TErrors,
                       strict: bool) -> TPinyinResult:
        """单字拼音转换.
        :param han: 单个汉字
        :param errors: 指定如何处理没有拼音的字符，详情请参考
                       :py:func:`~pypinyin.pinyin`
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :return: 返回拼音列表，多音字会有多个拼音项
        :rtype: list
        """
        num = ord(han)
        # 处理没有拼音的字符
        if num not in PINYIN_DICT:
            return self.handle_nopinyin(
                han,
                style=style,
                errors=errors,
                heteronym=heteronym,
                strict=strict)

        pys = PINYIN_DICT[num].split(',')  # 字的拼音列表

        post_data = self.post_pinyin(han, heteronym, [pys])
        if post_data is not None:
            pys = post_data[0]

        if not heteronym:
            orig_pinyin = pys[0]
            return [[
                self.convert_style(
                    han, orig_pinyin, style=style, strict=strict)
            ]]

        # 输出多音字的多个读音
        # 临时存储已存在的拼音，避免多音字拼音转换为非声调风格出现重复。
        # TODO: change to use set
        # TODO: add test for cache
        py_cached = {}
        pinyins = []
        for orig_pinyin in pys:
            py = self.convert_style(
                han, orig_pinyin, style=style, strict=strict)
            if py in py_cached:
                continue

            py_cached[py] = py
            pinyins.append(py)
        return [pinyins]

    def _phrase_pinyin(self,
                       phrase: Text,
                       style: TStyle,
                       heteronym: bool,
                       errors: TErrors,
                       strict: bool) -> TPinyinResult:
        """词语拼音转换.
        :param phrase: 词语
        :param errors: 指定如何处理没有拼音的字符
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :return: 拼音列表
        :rtype: list
        """
        py = []
        if phrase in PHRASES_DICT:
            py = deepcopy(PHRASES_DICT[phrase])

            post_data = self.post_pinyin(phrase, heteronym, py)
            if post_data is not None:
                py = post_data

            for idx, item in enumerate(py):
                han = phrase[idx]
                if heteronym:
                    py[idx] = _remove_dup_items([
                        self.convert_style(
                            han, orig_pinyin=x, style=style, strict=strict)
                        for x in item
                    ])
                else:
                    orig_pinyin = item[0]
                    py[idx] = [
                        self.convert_style(
                            han,
                            orig_pinyin=orig_pinyin,
                            style=style,
                            strict=strict)
                    ]
        else:
            for i in phrase:
                single = self._single_pinyin(
                    i,
                    style=style,
                    heteronym=heteronym,
                    errors=errors,
                    strict=strict)
                if single:
                    py.extend(single)
        return py

    def convert(self,
                words: Text,
                style: TStyle,
                heteronym: bool,
                errors: TErrors,
                strict: bool=...,
                **kwargs: Any) -> TPinyinResult:
        """根据参数把汉字转成相应风格的拼音结果。
        :param words: 汉字字符串
        :type words: unicode
        :param style: 拼音风格
        :param heteronym: 是否启用多音字
        :type heteronym: bool
        :param errors: 如果处理没有拼音的字符
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :type strict: bool
        :return: 按风格转换后的拼音结果
        :rtype: list
        """
        pys = []
        # 初步过滤没有拼音的字符
        if RE_HANS.match(words):
            pys = self._phrase_pinyin(
                words,
                style=style,
                heteronym=heteronym,
                errors=errors,
                strict=strict)
            return pys

        py = self.handle_nopinyin(
            words,
            style=style,
            errors=errors,
            heteronym=heteronym,
            strict=strict)

        if py:
            pys.extend(py)

        return pys

    def pre_convert_style(self,
                          han: Text,
                          orig_pinyin: Text,
                          style: TStyle,
                          strict: bool,
                          **kwargs: Any) -> Optional[Text]:
        """在把原始带声调的拼音按拼音风格转换前会调用 ``pre_convert_style`` 方法。
        如果返回值不为 ``None`` 会使用返回的结果代替 ``orig_pinyin``
        来进行后面的风格转换。
        :param han: 要处理的汉字
        :param orig_pinyin: 汉字对应的原始带声调拼音
        :param style: 要转换的拼音风格
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :param kwargs: 其他关键字参数，暂时无用，用于以后扩展新的参数。
        :return: ``None`` 或代替 ``orig_pinyin`` 参与拼音风格转换的拼音字符串。
        """
        pass

    def post_convert_style(self,
                           han: Text,
                           orig_pinyin: Text,
                           converted_pinyin: Text,
                           style: TStyle,
                           strict: bool,
                           **kwargs: Any) -> Optional[Text]:
        """在把原始带声调的拼音按拼音风格转换前会调用 ``pre_convert_style`` 方法。
        如果返回值不为 ``None`` 会使用返回的结果代替 ``converted_pinyin``
        作为拼音风格转换后的最终拼音结果。
        :param han: 要处理的汉字
        :param orig_pinyin: 汉字对应的原始带声调拼音
        :param converted_pinyin: 按拼音风格转换处理后的拼音
        :param style: 要转换的拼音风格
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :param kwargs: 其他关键字参数，暂时无用，用于以后扩展新的参数。
        :return: ``None`` 或代替 ``converted_pinyin`` 作为拼音风格转换后的拼音结果。
        """
        pass

    def _convert_style(self,
                       han: Text,
                       pinyin: Text,
                       style: TStyle,
                       strict: bool,
                       default: Text,
                       **kwargs: Any) -> Text:
        return convert_style(pinyin, style, strict, default=default, **kwargs)

    def convert_style(self,
                      han: Text,
                      orig_pinyin: Text,
                      style: TStyle,
                      strict: bool,
                      **kwargs: Any) -> Text:
        """按 ``style`` 的值对 ``orig_pinyin`` 进行处理，返回处理后的拼音
        转换风格前会调用 ``pre_convert_style`` 方法，
        转换后会调用 ``post_convert_style`` 方法。
        :param han: 要处理的汉字
        :param orig_pinyin: 汉字对应的原始带声调拼音
        :param style: 拼音风格
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :param kwargs: 其他关键字参数，暂时无用，用于以后扩展新的参数。
        :return: 按拼音风格转换处理后的拼音
        """
        pre_data = self.pre_convert_style(
            han, orig_pinyin, style=style, strict=strict)

        if pre_data is not None:
            pinyin = pre_data
        else:
            pinyin = orig_pinyin

        converted_pinyin = self._convert_style(
            han, pinyin, style=style, strict=strict, default=pinyin)

        post_data = self.post_convert_style(
            han, pinyin, converted_pinyin, style=style, strict=strict)
        if post_data is None:
            post_data = converted_pinyin

        return post_data

    def pre_handle_nopinyin(self,
                            chars: Text,
                            style: TStyle,
                            heteronym: bool,
                            errors: TErrors,
                            strict: bool) -> TNoPinyinResult:
        """处理没有拼音的字符串前会调用 ``pre_handle_nopinyin`` 方法。
        如果返回值不为 ``None`` 会使用返回的结果作为处理没有拼音字符串的结果，
        不再使用内置方法进行处理。
        :param chars: 待处理的没有拼音的字符串
        :param errors: 如何处理
        :param heteronym: 是否需要处理多音字
        :param kwargs: 其他关键字参数，暂时无用，用于以后扩展新的参数。
        :return: ``None`` 或代替 ``chars`` 参与拼音风格转换的拼音字符串
                  或拼音结果 list。
        """
        pass

    def post_handle_nopinyin(self,
                             chars: Text,
                             style: Style,
                             heteronym: bool,
                             errors: TErrors,
                             strict: bool,
                             pinyin: TNoPinyinResult,
                             **kwargs: Any) -> TNoPinyinResult:
        """处理完没有拼音的字符串后会调用 ``post_handle_nopinyin`` 方法。
        如果返回值不为 ``None`` 会使用返回的结果作为处理没有拼音的字符串的结果。
        :param chars: 待处理的没有拼音的字符串
        :param errors: 如何处理
        :param heteronym: 是否需要处理多音字
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :param pinyin: 处理后的拼音信息，值为空 list 或包含拼音信息的 list
        :param kwargs: 其他关键字参数，暂时无用，用于以后扩展新的参数。
        :return: ``None`` 或代替 ``pinyin`` 做为处理结果。
        """
        pass

    def _convert_nopinyin_chars(self,
                                chars: Text,
                                style: TStyle,
                                heteronym: bool,
                                errors: TErrors,
                                strict: bool) -> TNoPinyinResult:
        """转换没有拼音的字符。
        """
        if callable(errors):
            return errors(chars)

        if errors == 'default':
            return chars
        elif errors == 'ignore':
            return None
        elif errors == 'replace':
            if len(chars) > 1:
                return ''.join(str('%x' % ord(x)) for x in chars)
            else:
                return str('%x' % ord(chars))

    def handle_nopinyin(self,
                        chars: Text,
                        style: TStyle,
                        heteronym: bool,
                        errors: TErrors,
                        strict: bool,
                        **kwargs: Any) -> TPinyinResult:
        """处理没有拼音的字符串。
        处理前会调用 ``pre_handle_nopinyin`` 方法，
        处理后会调用 ``post_handle_nopinyin`` 方法。
        :param chars: 待处理的没有拼音的字符串
        :param style: 拼音风格
        :param errors: 如何处理
        :param heteronym: 是否需要处理多音字
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :return: 处理后的拼音结果，如果为 ``None`` 或空 list 表示忽略这个字符串.
        :rtype: list
        """
        pre_data = self.pre_handle_nopinyin(
            chars, style, errors=errors, heteronym=heteronym, strict=strict)

        if pre_data is not None:
            py = pre_data
        else:
            pre_data = chars
            py = self._convert_nopinyin_chars(
                pre_data,
                style,
                errors=errors,
                heteronym=heteronym,
                strict=strict)

        post_data = self.post_handle_nopinyin(
            chars,
            style,
            errors=errors,
            heteronym=heteronym,
            strict=strict,
            pinyin=py)
        if post_data is not None:
            py = post_data

        if not py:
            return []

        if isinstance(py, list):
            # 包含多音字信息
            if isinstance(py[0], list):
                if heteronym:
                    return py
                # [[a, b], [c, d]]
                # [[a], [c]]
                return [[x[0]] for x in py]

            return [[i] for i in py]
        else:
            return [[py]]


class _v2UConverter(V2UMixin, DefaultConverter):
    pass


class _neutralToneWith5Converter(NeutralToneWith5Mixin, DefaultConverter):
    pass


class _neutralToneWith5AndV2UConverter(NeutralToneWith5Mixin, V2UMixin,
                                       DefaultConverter):
    pass


class _mixConverter(DefaultConverter):
    def __init__(self, v_to_u=False, neutral_tone_with_five=False, **kwargs):
        super().__init__(**kwargs)
        self._v_to_u = v_to_u
        self._neutral_tone_with_five = neutral_tone_with_five

        self._v2uconverter = _v2UConverter()
        self._neutraltonewith5converter = _neutralToneWith5Converter()
        self._neutraltonewith5andv2uconverter = \
            _neutralToneWith5AndV2UConverter()

    def post_convert_style(self, han, orig_pinyin, converted_pinyin, style,
                           strict, **kwargs):
        if self._v_to_u and not self._neutral_tone_with_five:
            return self._v2uconverter.post_convert_style(
                han, orig_pinyin, converted_pinyin, style, strict, **kwargs)

        if self._neutral_tone_with_five and not self._v_to_u:
            return self._neutraltonewith5converter.post_convert_style(
                han, orig_pinyin, converted_pinyin, style, strict, **kwargs)

        if self._neutral_tone_with_five and self._v_to_u:
            return self._neutraltonewith5andv2uconverter.post_convert_style(
                han, orig_pinyin, converted_pinyin, style, strict, **kwargs)

        return super().post_convert_style(han, orig_pinyin, converted_pinyin,
                                          style, strict, **kwargs)
