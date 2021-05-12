from functools import wraps
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Text
from typing import Union

from pypinyin.constants import Style

TStyle = Style
TRegisterFunc = Optional[Callable[[Text, Dict[Any, Any]], Text]]
TWrapperFunc = Optional[Callable[[Text, Dict[Any, Any]], Text]]

# 存储各拼音风格对应的实现
_registry = {}  # type: Dict[Union[TStyle, int, str, Any], TRegisterFunc]


def convert(pinyin: Text,
            style: TStyle,
            strict: bool,
            default: Optional[Text]=None,
            **kwargs: Any) -> Text:
    """根据拼音风格把原始拼音转换为不同的格式
    :param pinyin: 原始有声调的单个拼音
    :type pinyin: unicode
    :param style: 拼音风格
    :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                   是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :type strict: bool
    :param default: 拼音风格对应的实现不存在时返回的默认值
    :return: 按照拼音风格进行处理过后的拼音字符串
    :rtype: unicode
    """
    if style in _registry:
        return _registry[style](pinyin, strict=strict, **kwargs)
    return default


def register(style: Union[TStyle, int, str, Any],
             func: TRegisterFunc=None) -> TWrapperFunc:
    """注册一个拼音风格实现
    ::
        @register('echo')
        def echo(pinyin, **kwargs):
            return pinyin
        # or
        register('echo', echo)
    """
    if func is not None:
        _registry[style] = func
        return

    def decorator(func):
        _registry[style] = func

        @wraps(func)
        def wrapper(pinyin, **kwargs):
            return func(pinyin, **kwargs)

        return wrapper

    return decorator


def auto_discover() -> None:
    """自动注册内置的拼音风格实现"""
    from pypinyin.style import (
        initials,
        tone,
        finals,
        bopomofo,
        cyrillic,
        others, )
