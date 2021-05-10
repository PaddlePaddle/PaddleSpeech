from pytest import mark

from pypinyin.contrib._tone_rule import right_mark_index

# http://www.hwjyw.com/resource/content/2010/06/04/8183.shtml


@mark.parametrize(
    'input,expected',
    [['da', 1], ['shuai', 3], ['guang', 2], ['zai', 1], ['po', 1], ['tou', 1],
     ['qiong', 2], ['ge', 1], ['jie', 2], ['teng', 1], ['yue', 2], ['lüe', 2],
     ['lve', 2], ['ji', 1], ['qing', 1], ['hu', 1], ['lü', 1], ['zhi', 2],
     ['chi', 2], ['hui', 2], ['qiu', 2], ['n', 0], ['ng', 0], ['m', 0],
     ['ê', 0], ['233', None]])
def test_right_mark_index(input, expected):
    assert right_mark_index(input) == expected
