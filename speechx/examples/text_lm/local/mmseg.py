#!/usr/bin/env python3
# modify from https://sites.google.com/site/homepageoffuyanwei/Home/remarksandexcellentdiscussion/page-2


class Word:
    def __init__(self, text='', freq=0):
        self.text = text
        self.freq = freq
        self.length = len(text)


class Chunk:
    def __init__(self, w1, w2=None, w3=None):
        self.words = []
        self.words.append(w1)
        if w2:
            self.words.append(w2)
        if w3:
            self.words.append(w3)

    #计算chunk的总长度
    def totalWordLength(self):
        length = 0
        for word in self.words:
            length += len(word.text)
        return length

    #计算平均长度
    def averageWordLength(self):
        return float(self.totalWordLength()) / float(len(self.words))

    #计算标准差
    def standardDeviation(self):
        average = self.averageWordLength()
        sum = 0.0
        for word in self.words:
            tmp = (len(word.text) - average)
            sum += float(tmp) * float(tmp)
        return sum

    #自由语素度
    def wordFrequency(self):
        sum = 0
        for word in self.words:
            sum += word.freq
        return sum


class ComplexCompare:
    def takeHightest(self, chunks, comparator):
        i = 1
        for j in range(1, len(chunks)):
            rlt = comparator(chunks[j], chunks[0])
            if rlt > 0:
                i = 0
            if rlt >= 0:
                chunks[i], chunks[j] = chunks[j], chunks[i]
                i += 1
        return chunks[0:i]

    #以下四个函数是mmseg算法的四种过滤原则，核心算法
    def mmFilter(self, chunks):
        def comparator(a, b):
            return a.totalWordLength() - b.totalWordLength()

        return self.takeHightest(chunks, comparator)

    def lawlFilter(self, chunks):
        def comparator(a, b):
            return a.averageWordLength() - b.averageWordLength()

        return self.takeHightest(chunks, comparator)

    def svmlFilter(self, chunks):
        def comparator(a, b):
            return b.standardDeviation() - a.standardDeviation()

        return self.takeHightest(chunks, comparator)

    def logFreqFilter(self, chunks):
        def comparator(a, b):
            return a.wordFrequency() - b.wordFrequency()

        return self.takeHightest(chunks, comparator)


#加载词组字典和字符字典
dictWord = {}
maxWordLength = 0


def loadDictChars(filepath):
    global maxWordLength
    fsock = open(filepath)
    for line in fsock:
        freq, word = line.split()
        word = word.strip()
        dictWord[word] = (len(word), int(freq))
        maxWordLength = len(word) if maxWordLength < len(
            word) else maxWordLength
    fsock.close()


def loadDictWords(filepath):
    global maxWordLength
    fsock = open(filepath)
    for line in fsock.readlines():
        word = line.strip()
        dictWord[word] = (len(word), 0)
        maxWordLength = len(word) if maxWordLength < len(
            word) else maxWordLength
    fsock.close()


#判断该词word是否在字典dictWord中
def getDictWord(word):
    result = dictWord.get(word)
    if result:
        return Word(word, result[1])
    return None


#开始加载字典
def run():
    from os.path import join, dirname
    loadDictChars(join(dirname(__file__), 'data', 'chars.dic'))
    loadDictWords(join(dirname(__file__), 'data', 'words.dic'))


class Analysis:
    def __init__(self, text):
        self.text = text
        self.cacheSize = 3
        self.pos = 0
        self.textLength = len(self.text)
        self.cache = []
        self.cacheIndex = 0
        self.complexCompare = ComplexCompare()

        #简单小技巧，用到个缓存，不知道具体有没有用处
        for i in range(self.cacheSize):
            self.cache.append([-1, Word()])

        #控制字典只加载一次
        if not dictWord:
            run()

    def __iter__(self):
        while True:
            token = self.getNextToken()
            if token is None:
                raise StopIteration
            yield token

    def getNextChar(self):
        return self.text[self.pos]

    #判断该字符是否是中文字符（不包括中文标点）
    def isChineseChar(self, charater):
        return 0x4e00 <= ord(charater) < 0x9fa6

    #判断是否是ASCII码
    def isASCIIChar(self, ch):
        import string
        if ch in string.whitespace:
            return False
        if ch in string.punctuation:
            return False
        return ch in string.printable

    #得到下一个切割结果
    def getNextToken(self):
        while self.pos < self.textLength:
            if self.isChineseChar(self.getNextChar()):
                token = self.getChineseWords()
            else:
                token = self.getASCIIWords() + '/'
            if len(token) > 0:
                return token
        return None

    #切割出非中文词
    def getASCIIWords(self):
        # Skip pre-word whitespaces and punctuations
        #跳过中英文标点和空格
        while self.pos < self.textLength:
            ch = self.getNextChar()
            if self.isASCIIChar(ch) or self.isChineseChar(ch):
                break
            self.pos += 1
        #得到英文单词的起始位置
        start = self.pos

        #找出英文单词的结束位置
        while self.pos < self.textLength:
            ch = self.getNextChar()
            if not self.isASCIIChar(ch):
                break
            self.pos += 1
        end = self.pos

        #Skip chinese word whitespaces and punctuations
        #跳过中英文标点和空格
        while self.pos < self.textLength:
            ch = self.getNextChar()
            if self.isASCIIChar(ch) or self.isChineseChar(ch):
                break
            self.pos += 1

        #返回英文单词
        return self.text[start:end]

    #切割出中文词，并且做处理，用上述4种方法
    def getChineseWords(self):
        chunks = self.createChunks()
        if len(chunks) > 1:
            chunks = self.complexCompare.mmFilter(chunks)
        if len(chunks) > 1:
            chunks = self.complexCompare.lawlFilter(chunks)
        if len(chunks) > 1:
            chunks = self.complexCompare.svmlFilter(chunks)
        if len(chunks) > 1:
            chunks = self.complexCompare.logFreqFilter(chunks)
        if len(chunks) == 0:
            return ''

        #最后只有一种切割方法
        word = chunks[0].words
        token = ""
        length = 0
        for x in word:
            if x.length != -1:
                token += x.text + "/"
                length += len(x.text)
        self.pos += length
        return token

    #三重循环来枚举切割方法，这里也可以运用递归来实现
    def createChunks(self):
        chunks = []
        originalPos = self.pos
        words1 = self.getMatchChineseWords()

        for word1 in words1:
            self.pos += len(word1.text)
            if self.pos < self.textLength:
                words2 = self.getMatchChineseWords()
                for word2 in words2:
                    self.pos += len(word2.text)
                    if self.pos < self.textLength:
                        words3 = self.getMatchChineseWords()
                        for word3 in words3:
                            # print(word3.length, word3.text)
                            if word3.length == -1:
                                chunk = Chunk(word1, word2)
                                # print("Ture")
                            else:
                                chunk = Chunk(word1, word2, word3)
                            chunks.append(chunk)
                    elif self.pos == self.textLength:
                        chunks.append(Chunk(word1, word2))
                    self.pos -= len(word2.text)
            elif self.pos == self.textLength:
                chunks.append(Chunk(word1))
            self.pos -= len(word1.text)

        self.pos = originalPos
        return chunks

    #运用正向最大匹配算法结合字典来切割中文文本
    def getMatchChineseWords(self):
        #use cache,check it
        for i in range(self.cacheSize):
            if self.cache[i][0] == self.pos:
                return self.cache[i][1]

        originalPos = self.pos
        words = []
        index = 0
        while self.pos < self.textLength:
            if index >= maxWordLength:
                break
            if not self.isChineseChar(self.getNextChar()):
                break
            self.pos += 1
            index += 1

            text = self.text[originalPos:self.pos]
            word = getDictWord(text)
            if word:
                words.append(word)

        self.pos = originalPos
        #没有词则放置个‘X’，将文本长度标记为-1
        if not words:
            word = Word()
            word.length = -1
            word.text = 'X'
            words.append(word)

        self.cache[self.cacheIndex] = (self.pos, words)
        self.cacheIndex += 1
        if self.cacheIndex >= self.cacheSize:
            self.cacheIndex = 0
        return words


if __name__ == "__main__":

    def cuttest(text):
        #cut =  Analysis(text)
        tmp = ""
        try:
            for word in iter(Analysis(text)):
                tmp += word
        except Exception as e:
            pass

        print(tmp)
        print("================================")

    cuttest(u"研究生命来源")
    cuttest(u"南京市长江大桥欢迎您")
    cuttest(u"请把手抬高一点儿")
    cuttest(u"长春市长春节致词。")
    cuttest(u"长春市长春药店。")
    cuttest(u"我的和服务必在明天做好。")
    cuttest(u"我发现有很多人喜欢他。")
    cuttest(u"我喜欢看电视剧大长今。")
    cuttest(u"半夜给拎起来陪看欧洲杯糊着两眼半晌没搞明白谁和谁踢。")
    cuttest(u"李智伟高高兴兴以及王晓薇出去玩，后来智伟和晓薇又单独去玩了。")
    cuttest(u"一次性交出去很多钱。 ")
    cuttest(u"这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。")
    cuttest(u"我不喜欢日本和服。")
    cuttest(u"雷猴回归人间。")
    cuttest(u"工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作")
    cuttest(u"我需要廉租房")
    cuttest(u"永和服装饰品有限公司")
    cuttest(u"我爱北京天安门")
    cuttest(u"abc")
    cuttest(u"隐马尔可夫")
    cuttest(u"雷猴是个好网站")
    cuttest(u"“Microsoft”一词由“MICROcomputer（微型计算机）”和“SOFTware（软件）”两部分组成")
    cuttest(u"草泥马和欺实马是今年的流行词汇")
    cuttest(u"伊藤洋华堂总府店")
    cuttest(u"中国科学院计算技术研究所")
    cuttest(u"罗密欧与朱丽叶")
    cuttest(u"我购买了道具和服装")
    cuttest(u"PS: 我觉得开源有一个好处，就是能够敦促自己不断改进，避免敞帚自珍")
    cuttest(u"湖北省石首市")
    cuttest(u"总经理完成了这件事情")
    cuttest(u"电脑修好了")
    cuttest(u"做好了这件事情就一了百了了")
    cuttest(u"人们审美的观点是不同的")
    cuttest(u"我们买了一个美的空调")
    cuttest(u"线程初始化时我们要注意")
    cuttest(u"一个分子是由好多原子组织成的")
    cuttest(u"祝你马到功成")
    cuttest(u"他掉进了无底洞里")
    cuttest(u"中国的首都是北京")
    cuttest(u"孙君意")
    cuttest(u"外交部发言人马朝旭")
    cuttest(u"领导人会议和第四届东亚峰会")
    cuttest(u"在过去的这五年")
    cuttest(u"还需要很长的路要走")
    cuttest(u"60周年首都阅兵")
    cuttest(u"你好人们审美的观点是不同的")
    cuttest(u"买水果然后来世博园")
    cuttest(u"买水果然后去世博园")
    cuttest(u"但是后来我才知道你是对的")
    cuttest(u"存在即合理")
    cuttest(u"的的的的的在的的的的就以和和和")
    cuttest(u"I love你，不以为耻，反以为rong")
    cuttest(u" ")
    cuttest(u"")
    cuttest(u"hello你好人们审美的观点是不同的")
    cuttest(u"很好但主要是基于网页形式")
    cuttest(u"hello你好人们审美的观点是不同的")
    cuttest(u"为什么我不能拥有想要的生活")
    cuttest(u"后来我才")
    cuttest(u"此次来中国是为了")
    cuttest(u"使用了它就可以解决一些问题")
    cuttest(u",使用了它就可以解决一些问题")
    cuttest(u"其实使用了它就可以解决一些问题")
    cuttest(u"好人使用了它就可以解决一些问题")
    cuttest(u"是因为和国家")
    cuttest(u"老年搜索还支持")
    cuttest(
        u"干脆就把那部蒙人的闲法给废了拉倒！RT @laoshipukong : 27日，全国人大常委会第三次审议侵权责任法草案，删除了有关医疗损害责任“举证倒置”的规定。在医患纠纷中本已处于弱势地位的消费者由此将陷入万劫不复的境地。 "
    )
    cuttest("2022年12月30日是星期几？")
    cuttest("二零二二年十二月三十日是星期几？")
