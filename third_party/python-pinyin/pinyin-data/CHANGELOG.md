# ChangeLog

## [0.10.2] (2021-03-13)

* 修改 `帧` 的最常用读音为 `zhēn`
* 修复 `zdic.txt` 中两个拼音字母 `è í` 使用不当的问题. Thanks [@Ace-Who](https://github.com/Ace-Who)


## [0.10.1] (2020-11-22)

* 调整 `地` 和 `謦` 的拼音顺序


## [0.10.0] (2020-10-07)

* 新增 `kTGHZ2013.txt`: [Unihan Database][unihan] 中 [kTGHZ2013](http://www.unicode.org/reports/tr38/#kTGHZ2013) 部分的拼音数据（来源于《通用规范汉字字典》的拼音数据）
* 修正部分拼音的读音
* 生成 `pinyin.txt` 时合并来自 `kTGHZ2013.txt` 的拼音数据


## [0.9.0] (2020-06-06)

* 更新 Unihan 数据版本为 13.0.0


## [0.8.1] (2019-10-26)

* 修正 `迹` 和 `分` 的读音。


## [0.8.0] (2019-06-01)

* 增加 `kanji.txt` 日本自造汉字的拼音数据 via [#32]. Thanks [@LuoZijun](https://github.com/LuoZijun)
* 去掉几个有误的轻声数据


## [0.7.0] (2019-03-31)

* 更新 Unihan 数据版本为 12.0.0


## [0.6.2] (2018-09-16)

* 修改 `蹒` 的最常用读音为 `pán`


## [0.6.1] (2018-08-04)

* 修改 `著` 的默认读音为 `zhù` via [8802f31]


## [0.6.0] (2018-07-08)

* 更新 Unihan 数据版本为 11.0.0 via [68dc169]


## [0.5.1] (2018-04-19)

* 更正 `卓`、`啥` 的拼音数据 via [#26] 。Thanks [@shibingli](https://github.com/shibingli)
* 更新 `〇` 的拼音数据 via [#27]


## [0.5.0] (2018-03-18)

* 更新 Unihan 数据版本为 10.0.0 via [#19][#19]
* 新增 kMandarin_overwrite.txt 用于手工纠正 kMandarin.txt 中有误的拼音数据 via [#21][#21]
* 更正 `讽`、`识` 的最常用读音 via [#20][#20]
* 更正 埔,彷,珖,U+275C8 的常用发音 [635b238c4](https://github.com/mozillazg/pinyin-data/commit/635b238c4d21e55d8fd66299c8da3ae555253b3a)


## [0.4.1] (2017-02-12)

* `妳` 的最常用拼音调整为 `nǐ` via [eb08200](https://github.com/mozillazg/pinyin-data/commit/eb08200d0a203c57ecc62ec7a118765518430238)
* `钭` 的拼音更新为 `tǒu,dǒu` via [fb9e64e](https://github.com/mozillazg/pinyin-data/commit/fb9e64e6c0a20eb0e792e8a402dffbf8cc2dfa57)


## [0.4.0] (2016-10-17)

* Update PUA.txt 详见 [#7](https://github.com/mozillazg/pinyin-data/issues/7) thanks [@Artoria2e5][@Artoria2e5]
* Rename PUA.txt to GBK_PUA.txt 详见 [#7](https://github.com/mozillazg/pinyin-data/issues/7)
* Add kMandarin_8105.txt (《通用规范汉字表》里 8105 个汉字最常用的一个读音) [#9][#9] [#11][#11]
* Update pinyin.txt with latest data


## [0.3.0] (2016-08-19)

* Fixed format of zdic.txt via [b8e4394](https://github.com/mozillazg/pinyin-data/commit/b8e439490d2c6e8c711652983db52fb69136919b).
* Fixed some pinyin: 罗 via [468ffaa](https://github.com/mozillazg/pinyin-data/commit/468ffaa8eb678637c7565a02e6836255bd0df06c).
* Support Chinese that in PUA([Private Use Area](https://en.wikipedia.org/wiki/Private_Use_Areas>)) via [#2](https://github.com/mozillazg/pinyin-data/pull/2).
* pinyin.txt add line comments that startswith `#` via [9944f79](https://github.com/mozillazg/pinyin-data/commit/9944f795e191fb3606d65ada84b6fad5665f8776).


## [0.2.0] (2016-07-19)

* Update to the latest version of [Unihan Database](http://www.unicode.org/charts/unihan.html):

  > Date: 2016-06-01 07:01:48 GMT [JHJ]       
  > Unicode version: 9.0.0


## 0.1.0 (2016-03-11)

* Initial Release


[@Artoria2e5]: https://github.com/Artoria2e5
[#9]: https://github.com/mozillazg/pinyin-data/pull/9
[#11]: https://github.com/mozillazg/pinyin-data/pull/11
[#19]: https://github.com/mozillazg/pinyin-data/pull/19
[#20]: https://github.com/mozillazg/pinyin-data/pull/20
[#21]: https://github.com/mozillazg/pinyin-data/pull/21
[#26]: https://github.com/mozillazg/pinyin-data/pull/26
[#27]: https://github.com/mozillazg/pinyin-data/pull/27
[68dc169]: https://github.com/mozillazg/pinyin-data/commit/68dc169c3f0f02cb9bf53290edab2d2d2463e0c5
[8802f31]: https://github.com/mozillazg/pinyin-data/commit/8802f31e0e65c6e34a497adb55993425741a9d41
[#32]: https://github.com/mozillazg/pinyin-data/pull/32
[unihan]: http://www.unicode.org/charts/unihan.html

[0.2.0]: https://github.com/mozillazg/pinyin-data/compare/v0.1.0...v0.2.0
[0.3.0]: https://github.com/mozillazg/pinyin-data/compare/v0.2.0...v0.3.0
[0.4.0]: https://github.com/mozillazg/pinyin-data/compare/v0.3.0...v0.4.0
[0.4.1]: https://github.com/mozillazg/pinyin-data/compare/v0.4.0...v0.4.1
[0.5.0]: https://github.com/mozillazg/pinyin-data/compare/v0.4.1...v0.5.0
[0.5.1]: https://github.com/mozillazg/pinyin-data/compare/v0.5.0...v0.5.1
[0.6.0]: https://github.com/mozillazg/pinyin-data/compare/v0.5.1...v0.6.0
[0.6.1]: https://github.com/mozillazg/pinyin-data/compare/v0.6.0...v0.6.1
[0.6.2]: https://github.com/mozillazg/pinyin-data/compare/v0.6.1...v0.6.2
[0.7.0]: https://github.com/mozillazg/pinyin-data/compare/v0.6.2...v0.7.0
[0.8.0]: https://github.com/mozillazg/pinyin-data/compare/v0.7.0...v0.8.0
[0.8.1]: https://github.com/mozillazg/pinyin-data/compare/v0.8.0...v0.8.1
[0.9.0]: https://github.com/mozillazg/pinyin-data/compare/v0.8.1...v0.9.0
[0.10.0]: https://github.com/mozillazg/pinyin-data/compare/v0.9.0...v0.10.0
[0.10.1]: https://github.com/mozillazg/pinyin-data/compare/v0.10.0...v0.10.1
[0.10.2]: https://github.com/mozillazg/pinyin-data/compare/v0.10.1...v0.10.2
