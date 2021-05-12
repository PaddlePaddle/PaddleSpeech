pymmseg-cpp

* by pluskid & kronuz
* http://github.com/pluskid/pymmseg-cpp

# DESCRIPTION:

pymmseg-cpp is a Python interface to rmmseg-cpp. rmmseg-cpp is a high
performance Chinese word segmentation utility for Ruby. However, the
core part is written in C++ independent of Ruby. So I decide to write
a Python interface for it in order to use it in my Python project.

# FEATURES:

* Runs fast and the memory consumption is small.
* Support user customized dictionaries.
* UTF-8 and Unicode encoding is supported.

# SYNOPSIS:

## A simple script

pymmseg-cpp provides a simple script (bin/pymmseg), which can read the
text from standard input and print the segmented result to standard
output. Try pymmseg -h for help on the options.

## As a Python module

To use pymmseg-cpp in normal Python program, first import the module and
init by loading the dictionaries:

```python
import mmseg

mmseg.Dictionary.load_dictionaries()
```

If you want to load your own customized dictionaries, please customize
`mmseg.Dictionary.dictionaries` before calling load_dictionaries.

Then create an Algorithm iterable object and iterate through it:

```python
algor = mmseg.Algorithm(text)
for tok in algor:
    print '%s [%d..%d]' % (tok.text, tok.start, tok.end)
```

## Customize the dictionary

You can also load your own character dictionary or word dictionary in the
following way:

```python
import mmseg
mmseg.Dictionary.load_words('customize_words.dic')
mmseg.Dictionary.load_chars('customize_chars.dic')
```

### Format for chars.dic

* each line contains the freq of the character, a space, and then the character

### Format for words.dic

* each line contains the length of the word, a space, and then the word

### WARNING

* The length of the word means number of characters in the word, not number of bytes
* The format of words.dic is different from chars.dic, see above
* There should be a newline at the end of all the dict file

# REQUIREMENTS:

* python 2.5+
* g++

# INSTALLATION:

pymmseg-cpp should be installed using pip:

```
pip install pymmseg (instead of pymmseg-cpp, see below)
```

or setuptools:

```
easy_install pymmseg
```

You can also download the latest code from github and build it yourself:

```
python setup.py build
```

Then copy the pymmseg directory to your Python's package path. e.g.
`/usr/lib/python2.5/site-packages/`. Now you can use pymmseg in your
application.

# Alternative Version

There is a package called `pymmseg-cpp` in PyPI. That is a modified version by Shenpeng Zhang (zsp007@gmail.com) based on an earlier version of this project. The version number in those two packages are independent. The naming is a little confusing, and unfortunately both of us don't have enough time to get the changes merged properly. I'll list the known differences here so that you can choose which version to use:

* pymmseg is using Python native extension code (instead of the original interface based on ctypes) with the help of Kronuz, who claimed ~400% performance boost.
* pymmseg-cpp has a refined built-in dictionary file (EDIT: Now also incorporated in pymmseg)
* pymmseg-cpp ships with some helper functions that might be convenient when using with xapian

# CONTRIBUTIONS:

Python native extension code contributed by German M. Bravo (Kronuz)
for a ~400% performance boost under Python.

# LICENSE:

(The MIT License)

Copyright (c) 2012

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
