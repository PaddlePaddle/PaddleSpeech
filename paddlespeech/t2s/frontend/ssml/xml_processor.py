# -*- coding: utf-8 -*-
import re
import xml.dom.minidom
import xml.parsers.expat
from xml.dom.minidom import Node
from xml.dom.minidom import parseString
'''
Note:  xml 有5种特殊字符， &<>"'
其一，采用<![CDATA[ ]]>特殊标签，将包含特殊字符的字符串封装起来。
例如：
<TitleName><![CDATA["姓名"]]></TitleName>
其二，使用XML转义序列表示这些特殊的字符，这5个特殊字符所对应XML转义序列为：
&  &amp;
<  &lt;
>  &gt;
"  &quot;
'  &apos;
例如：
<TitleName>&quot;姓名&quot;</TitleName>
'''


class MixTextProcessor():
    def __repr__(self):
        print("@an MixTextProcessor class")

    def get_xml_content(self, mixstr):
        '''返回字符串的 xml 内容'''
        xmlptn = re.compile(r"<speak>.*?</speak>", re.M | re.S)
        ctn = re.search(xmlptn, mixstr)
        if ctn:
            return ctn.group(0)
        else:
            return None

    def get_content_split(self, mixstr):
        ''' 文本分解，顺序加了列表中，按非 xml 和 xml 分开，对应的字符串,带标点符号
        不能去除空格，因为 xml 中tag 属性带空格
        '''
        ctlist = []
        # print("Testing:",mixstr[:20])
        patn = re.compile(r'(.*\s*?)(<speak>.*?</speak>)(.*\s*)$', re.M | re.S)
        mat = re.match(patn, mixstr)
        if mat:
            pre_xml = mat.group(1)
            in_xml = mat.group(2)
            after_xml = mat.group(3)

            ctlist.append(pre_xml)
            ctlist.append(in_xml)
            ctlist.append(after_xml)
            return ctlist
        else:
            ctlist.append(mixstr)
        return ctlist

    @classmethod
    def get_pinyin_split(self, mixstr):
        ctlist = []
        patn = re.compile(r'(.*\s*?)(<speak>.*?</speak>)(.*\s*)$', re.M | re.S)
        mat = re.match(patn, mixstr)
        if mat:
            # pre <speak>
            pre_xml = mat.group(1)
            # between <speak> ... </speak>
            in_xml = mat.group(2)
            # post </speak>
            after_xml = mat.group(3)

            # pre with none syllable
            ctlist.append([pre_xml, []])

            # between with syllable
            # [(sub sentence, [syllables]), ...]
            dom = DomXml(in_xml)
            pinyinlist = dom.get_pinyins_for_xml()
            ctlist = ctlist + pinyinlist

            # post with none syllable
            ctlist.append([after_xml, []])
        else:
            ctlist.append([mixstr, []])
        return ctlist

    @classmethod
    def get_dom_split(self, mixstr):
        ''' 文本分解，顺序加了列表中，返回文本和say-as标签
        '''
        ctlist = []
        patn = re.compile(r'(.*\s*?)(<speak>.*?</speak>)(.*\s*)$', re.M | re.S)
        mat = re.match(patn, mixstr)
        if mat:
            pre_xml = mat.group(1)
            in_xml = mat.group(2)
            after_xml = mat.group(3)

            ctlist.append(pre_xml)
            dom = DomXml(in_xml)
            tags = dom.get_text_and_sayas_tags()
            ctlist.extend(tags)

            ctlist.append(after_xml)
            return ctlist
        else:
            ctlist.append(mixstr)
        return ctlist


class DomXml():
    def __init__(self, xmlstr):
        self.tdom = parseString(xmlstr)  #Document
        self.root = self.tdom.documentElement  #Element
        self.rnode = self.tdom.childNodes  #NodeList

    def get_text(self):
        '''返回 xml 内容的所有文本内容的列表'''
        res = []

        for x1 in self.rnode:
            if x1.nodeType == Node.TEXT_NODE:
                res.append(x1.value)
            else:
                for x2 in x1.childNodes:
                    if isinstance(x2, xml.dom.minidom.Text):
                        res.append(x2.data)
                    else:
                        for x3 in x2.childNodes:
                            if isinstance(x3, xml.dom.minidom.Text):
                                res.append(x3.data)
                            else:
                                print("len(nodes of x3):", len(x3.childNodes))

        return res

    def get_xmlchild_list(self):
        '''返回 xml 内容的列表，包括所有文本内容(不带 tag)'''
        res = []

        for x1 in self.rnode:
            if x1.nodeType == Node.TEXT_NODE:
                res.append(x1.value)
            else:
                for x2 in x1.childNodes:
                    if isinstance(x2, xml.dom.minidom.Text):
                        res.append(x2.data)
                    else:
                        for x3 in x2.childNodes:
                            if isinstance(x3, xml.dom.minidom.Text):
                                res.append(x3.data)
                            else:
                                print("len(nodes of x3):", len(x3.childNodes))
        print(res)
        return res

    def get_pinyins_for_xml(self):
        '''返回 xml 内容，字符串和拼音的 list '''
        res = []

        for x1 in self.rnode:
            if x1.nodeType == Node.TEXT_NODE:
                t = re.sub(r"\s+", "", x1.value)
                res.append([t, []])
            else:
                for x2 in x1.childNodes:
                    if isinstance(x2, xml.dom.minidom.Text):
                        t = re.sub(r"\s+", "", x2.data)
                        res.append([t, []])
                    else:
                        # print("x2",x2,x2.tagName)
                        if x2.hasAttribute('pinyin'):
                            pinyin_value = x2.getAttribute("pinyin")
                            pinyins = pinyin_value.split(" ")
                        for x3 in x2.childNodes:
                            # print('x3',x3)
                            if isinstance(x3, xml.dom.minidom.Text):
                                t = re.sub(r"\s+", "", x3.data)
                                res.append([t, pinyins])
                            else:
                                print("len(nodes of x3):", len(x3.childNodes))

        return res

    def get_all_tags(self, tag_name):
        '''获取所有的 tag 及属性值'''
        alltags = self.root.getElementsByTagName(tag_name)
        for x in alltags:
            if x.hasAttribute('pinyin'):  # pinyin
                print(x.tagName, 'pinyin',
                      x.getAttribute('pinyin'), x.firstChild.data)

    def get_text_and_sayas_tags(self):
        '''返回 xml 内容的列表，包括所有文本内容和<say-as> tag'''
        res = []

        for x1 in self.rnode:
            if x1.nodeType == Node.TEXT_NODE:
                res.append(x1.value)
            else:
                for x2 in x1.childNodes:
                    res.append(x2.toxml())
        return res
