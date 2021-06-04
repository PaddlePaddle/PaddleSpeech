"""Traditional and simplified Chinese conversion with 
`opencc <https://github.com/BYVoid/OpenCC>`_.
"""


import opencc

_t2s_converter = opencc.OpenCC("t2s.json")
_s2t_converter = opencc.OpenCC('s2t.json')

def tranditional_to_simplified(text: str) -> str:
    return _t2s_converter.convert(text)

def simplified_to_traditional(text: str) -> str:
    return _s2t_converter.convert(text)
