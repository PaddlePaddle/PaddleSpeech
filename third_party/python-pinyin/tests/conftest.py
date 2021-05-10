import os
import pytest


@pytest.fixture(autouse=True, scope='function')
def setup():
    os.environ.pop('PYPINYIN_NO_PHRASES', None)
    os.environ.pop('PYPINYIN_NO_DICT_COPY', None)
    try:
        yield
    finally:
        os.environ.pop('PYPINYIN_NO_PHRASES', None)
        os.environ.pop('PYPINYIN_NO_DICT_COPY', None)
