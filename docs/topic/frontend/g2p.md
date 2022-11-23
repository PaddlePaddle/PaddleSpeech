# g2p 字典设计
<!--
modified from https://zhuanlan.zhihu.com/p/349600439
-->
本文主要讲语音合成的 g2p (grapheme to phoneme) 部分。

代码: [generate_lexicon.py](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/other/mfa/local/generate_lexicon.py) （代码可能与此处的描述有些许出入，以代码为准，生成的带 tone 带儿化的 pinyin 字典参考 [simple.lexicon](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/csmsc/tts3/local/simple.lexicon)）

## ARPAbet
对于英文 TTS，常用的 g2p 是通过查询 CMUDict 来实现，而 CMUDict 注音使用的系统是 ARPAbet，具体含义参见 [CMU 发音词典](http://www.speech.cs.cmu.edu/cgi-bin/cmudict/)。

它包含 39 个 phoneme， 不包含音词汇重音的变体:

| Phoneme       | Example | Translation |
|:-------------:|:-------:|:-----------:|
|     AA        |  odd    |    AA D     |
|     AE        |  at     |    AE T     |
|     AH        |  hut    |    HH AH T  |
|     AO        |  ought  |    AO T     |
|     AW        |  cow    |    K AW     |
|     AY        |  hide   |    HH AY D  |
|     B         |  be     |    B IY     |
|     CH        |  cheese |    CH IY Z  |
|     D         |  dee    |    D IY     |
|     DH        |  thee   |    DH IY    |
|     EH        |  Ed     |    EH D     |
|     ER        |  hurt   |    HH ER T  |
|     EY        |  ate    |    EY T     |
|     F         |  fee    |    F IY     |
|     G         |  green  |    G R IY N |
|     HH        |  he     |    HH IY    |
|     IH        |  it     |    IH T     |
|     IY        |  eat    |    IY T     |
|     JH        |  gee    |    JH IY    |
|     K         |  key    |    K IY     |
|     L         |  lee    |    L IY     |
|     M         |  me     |    M IY     |
|     N         |  knee   |    N IY     |
|     NG        |  ping   |    P IH NG  |
|     OW        |  oat    |    OW T     |
|     OY        |  toy    |    T OY     |
|     P         |  pee    |    P IY     |
|     R         |  read   |    R IY D   |
|     S         |  sea    |    S IY     |
|     SH        |  she    |    SH IY    |
|     T         |  tea    |    T IY     |
|     TH        |  theta  |    TH EY T AH|
|     UH        |  hood   |    HH UH D  |
|     UW        |  two    |    T UW     |
|     V         |  vee    |    V IY     |
|     W         |  we     |    W IY     |
|     Y         |  yield  |    Y IY L D |
|     Z         |  zee    |    Z IY     |
|     ZH        |  seizure|    S IY ZH ER|

另外还包含三个重音标记，

0 — No stress
1 — Primary stress
2 — Secondary stress

其中重音标记附在元音后面。当只需要音标而不需要重音标记的时候也可以直接省略。

CMUDict 只是一个词典，当出现了不在词典中的词时（OOV），可以求助其他工具可以根据拼写得到对应的发音，如: 
  - [Lexicon Tool](http://www.speech.cs.cmu.edu/tools)
  - [g2p-seq2seq](https://github.com/cmusphinx/g2p-seq2seq)

## 中文注音系统

中文普通话的注音系统存在许多套，比如汉语拼音 (pinyin)， 注音符号 (bopomofo)， 国语注音符第二式， 威妥玛拼音等。而且有一些并非注音方案，是拉丁化方案，因此为了符号系统的经济性，会做一些互补符号的简并，比如汉语拼音中的 `i` 的代表了三个音位， `e` 代表了两个音位（单用的情况很少， 单用时写作 `ê`）；也有一些简写，比如 `bpmf` 后的 `o` 是 `uo` 的简写， `ui` 是 `uei` 的简写，` iu` 是 `iou` 的简写， `un` 是 `uen` 的简写， `ao` 是为了书写避免形近而改掉的 `au`， `y` 和 `w` 是为了连续书写时作为分隔而产生的零声母， `ü` 在 `j`、 `q`、 `x` 后面省略两点（中国大陆使用美式键盘打字的时候，一般只有在“女”、 “律”、“略”和“虐”这一类的字里面用 `v` 代替 `ü`，而在 `j`、 `q`、 `x` 后面的时候则仍用 `u` ），有鼻韵母 `uang` 而没有 `ueng`，但是又有 `weng` 这个音节之类的问题， 有 `ong` 韵母但是又没有单用的情形。其实这些都是汉语拼音作为拉丁化方案而做的一系列的修改。

另外，汉语的声调是用了特殊符号来标调型，用字母记录的时候常用 `12345` 或者 `1234`、轻音不标等手段。

另外还有两个比较突出的问题是**儿化**和**变调**（参考 [zh_text_frontend](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/tts/zh_text_frontend.md)）。对于具体的数据集，也可能有不同的标注方案。一般我们为汉字标音是标字调而不标变调，但是**标贝数据集是标变调的**（但是也没有把所有的变调都正确标出来）。儿化在汉语书写和拼音中也是一个比较麻烦的事情，虽然正字法中说到可以用小字号的儿表示儿化，但是这种发音由字号这种排版要素来表达的手法未免过于崎岖，所以鲜见有人真的这么排版，只有在某些书籍中，强调此事的时候见过。另外，在儿化的标音方式上，鼻韵母需要去掉韵尾然后换成 r，这么一来，如果直接抽取拼音的字符串表示，那么可能出现的音节就会超过 1400， 甚至进入一种含糊的状态，不清楚一共有多少个有效音节，即使是韵母，也会因此扩展近一倍。

因为存在这样的情形，再考虑到不同的数据集自带的拼音 transcription 的风格可能不同，所以需要考虑进行转换，在内部转成统一的表示。既然这个过程是必要的，那么我们可以大胆设计一个内部方案。

这里设计的原则是：

1. 有效符号集仅切分为声母和韵母，不作声母，介音，韵腹，韵尾的切分；

2. 尽可能把不同的音用不同的符号表示，比如 `i` 的 `e` 会被拆分为 3 和 2 个符号， `u` 和 `ü` 开头的韵母分开，这是为了 TTS 系统的建议性考虑的，我们选择尽量反映语音的现实情况，而不把注音系统里面的奇怪规则留给模型去学习；

3. 不包含零声母 `y`， `w`之类的形式上的符号，因为如果这些符号不发声或者发声极短，那么可以不加入音符序列中，以期待 attention 更对角；

4. 声调和韵母不结合为一个符号，而是分开，这样可以**减少词汇量**，使得符号的 embedding 得到更充分的训练，也更能反映声调语言的特点（数据集少时推荐这么做）；

5. 儿化的标音方式采用拆分的方式处理， 但是增设一个特殊符号 `&r` 来表示儿化的 `r`，它和一般的 `er` 不同，以区分实际读音的区别。

6. 更加贴近注音符号，把 `in` 写作 `ien`，`ing` 写作 `ieng`， `un` 写作 `uen`， `ong` 写作 `ueng`， `iong` 写作 `üeng`。其中 `in` 和 `ing` 的转写纯属偏好，无论用什么符号写，都可以被转为一个 index， 只要它们的使用情况不发声变化就可以。而 `ong` 写作 `ueng` 则是有实际差别的，如果 `ong` 作为一个韵母，那么 `weng` 经过修改之后会变成 `ueng`， 就会同时有 `ueng` 和 `ong`。而如果不细究音值上的微妙差异，`ong` 就是 `ung` 的一种奇怪表示， 在注意符号中， 它就记作 `ㄨㄥ`。而 `iong` 则是 `ㄩㄥ`。

7. `ui`， `iu` 都展开为 `uei` 和 `iou` ， 纯属偏好，对实际结果没有影响。`bpmf `后的 `o` 展开为 `uo`，这个则是为了和单独的 `o` 区分开（哦， 和波里面的韵母的发音其实不同）。

8. 所有的 `ü `都有 `v` 代替，无论是单独作韵母， 还是复韵母和鼻韵母。

9. 把停顿以 `#1` 等方式纳入其中， 把 `<pad>` `<unk>` `<s>` `</s>` 这些为了处理符号系列的特殊符号也加入其中，多一些特殊词汇并不会对 Embedding 产生什么影响。

于是我们可以的通过一套规则系统，把标贝的**拼音标注**转换成我们需要的形式。（当然，如果是别的数据集的实际标注不同，那么转换规则也要作一些修改)

在实际使用中文数据集时，我们仅使用其提供的**拼音标注**，而不使用**音素标注**（PhoneLabel），因为不同的数据集有不同的标注规则，而且有的数据集是没有**音素标注**的（如，aishell3）

我们的做法和维基百科上的汉语拼音音节列表更接近 [汉语拼音音节列表](https://zh.wikipedia.org/zh-hans/%E6%B1%89%E8%AF%AD%E6%8B%BC%E9%9F%B3%E9%9F%B3%E8%8A%82%E5%88%97%E8%A1%A8)

转换之后，符号列表是：

声母基本没有什么争议，共 21 个:
|声母|
|:--:|
|b|
|p|
|m|
|f|
|d|
|t|
|n|
|l|
|g|
|k|
|h|
|j|
|q|
|x|
|zh|
|ch|
|sh|
|r|
|z|
|c|
|s|

韵母和儿化韵尾（共 41个）
|韵母|解释|
|:----:|:-----------: |
|ii     |`zi`，`ci`， `si` 里面的韵母 `i`|
|iii    |`zhi`， `chi`， `shi`， `ri` 里面的韵母 `i`|
|a    |啊，卡|
|o    |哦|
|e    |恶，个|
|ea    |ê|
|ai    |爱，在|
|ei    |诶，薇|
|ao    |奥，脑|
|ou    |欧，勾|
|an    |安，单|
|en    |恩，痕|
|ang    |盎，刚|
|eng    |嗯，更|
|er    |儿|
|i    |一|
|ia    |鸦，家|
|io    |哟|
|ie    |叶，界|
|iai    |崖（台语发音）|
|iao    |要，教|
|iou    |有，久|
|ian    |言，眠|
|ien    |因，新|
|iang    |样，降|
|ieng    |英，晶
|u    |无，卢|
|ua    |哇，瓜|
|uo    |我，波|
|uai    |外，怪|
|uei    |位，贵|
|uan    |万，乱|
|uen    |问，论|
|uang   |网，光|
|ueng   |翁，共|
|v      |玉，曲，`ü`|
|ve     |月，却|
|van    |源，倦|
|ven    |韵，君|
|veng   |永，炯|
|&r     |儿化韵尾|
