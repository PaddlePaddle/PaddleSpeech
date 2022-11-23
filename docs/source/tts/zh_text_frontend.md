# Chinese Rule-Based Text Frontend
A TTS system mainly includes three modules: `Text Frontend`, `Acoustic model` and `Vocoder`. We provide a complete Chinese text frontend module in PaddleSpeech TTS, see exapmles in [examples/other/tn](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/tn) and [examples/other/g2p](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/g2p).

A text frontend module mainly includes:
 - Text Segmentation
 - Text Normalization (TN)
 - Word Segmentation (mainly in Chinese)
 - Part-of-Speech
 - Prosody
 - G2P (Grapheme-to-Phoneme, include Polyphone and Tone Sandhi, etc.)
 - Linguistic Features/Charactors/Phonemes

```text
• text: 90 后为中华人民共和国成立 70 周年准备了大礼
• Text Normalization: 九零后为中华人民共和国成立七十周年准备了大礼
• Word Segmentation: 九零后/为/中华人民/共和国/成立/七十/周年/准备/了/大礼
• G2P:
    jiu3 ling2 hou4 wei4 zhong1 hua2 ren2 min2 gong4 he2 guo2 ...
• Prosody (prosodic words #1, prosodic phrases #2, intonation phrases #3, sentence #4):
    九零后#1为中华人民#1共和国#2成立七十周年#3准备了大礼#4
```

Among them, Text Normalization and G2P are the most important modules. We mainly introduce them here.

## Text Normalization
### Supported NSW (Non-Standard-Word) Normalization

|NSW type|raw|normalized|
|:--|:-|:-|
|serial number|电影中梁朝伟扮演的陈永仁的编号27149|电影中梁朝伟扮演的陈永仁的编号二七一四九|
|cardinal|这块黄金重达324.75克<br>我们班的最高总分为583分|这块黄金重达三百二十四点七五克<br>我们班的最高总分为五百八十三分|
|numeric range |12\~23<br>-1.5\~2|十二到二十三<br>负一点五到二|
|date|她出生于86年8月18日，她弟弟出生于1995年3月1日|她出生于八六年八月十八日， 她弟弟出生于一九九五年三月一日|
|time|等会请在12:05请通知我|等会请在十二点零五分请通知我
|temperature|今天的最低气温达到-10°C|今天的最低气温达到零下十度
|fraction|现场有7/12的观众投出了赞成票|现场有十二分之七的观众投出了赞成票|
|percentage|明天有62％的概率降雨|明天有百分之六十二的概率降雨|
|money|随便来几个价格12块5，34.5元，20.1万|随便来几个价格十二块五，三十四点五元，二十点一万|
|telephone|这是固话0421-33441122<br>这是手机+86 18544139121|这是固话零四二一三三四四一一二二<br>这是手机八六一八五四四一三九一二一|


## Grapheme-to-Phoneme
In Chinese, G2P is a very complex module, which mainly includes **polyphone**  and **tone sandhi**.

We use [g2pM](https://github.com/kakaobrain/g2pM) and [pypinyin](https://github.com/mozillazg/python-pinyin)  as the default g2p tools. They can solve the problem of polyphones to a certain extent. In the future, we intend to use a trainable language model (for example, [BERT](https://arxiv.org/abs/1810.04805)) for polyphones.

However, g2pM and pypinyin do not perform well in tone sandhi, we use rules to solve this problem, which requires relevant linguistic knowledge.

The **tone sandhi** in Chinese mainly include:

 - soft tone sandhi (轻声变调)
 - "一" "不" tone sandhi ("一" "不" 变调)
 - three tone sandhi  (三声变调)

For ease of understanding, we list the tone sandhi rules in Chinese here
### 1. 轻声变调
|  |cases  |
|:--|:-|
| 语气助词“吧、呢、啊”等 | 吃吧、走吗、去呢、跑啊 |
| 结构助词：“的、地、得”| 我的书、慢慢地走、跑得很快等 |
|有的轻声音节和非轻声音节构成对比区别意义 |买卖：一指生意；二指买和卖。 <br/> 地道：一指纯粹、真正；二指地下通道。<br> 大意：一指没有注意；二指主要的意思。 <br/>  东西：一指各种事物；二指东面与西面。<br>  言语：一指所说的话；二指开口，招呼。<br/>运气：一指一种锻炼的方法。二指幸运。<br> |
|名词的后缀：“们、子、头”|你们、房子、石头 |
|名词或动词的第二个重叠音节 | 奶奶、姐姐、爸爸、试试、看看、说说、问问 |
|名词后面表示方位的：“上、下、里” |桌上、地下、院里  |
| 动态助词：“了、着、过” | 走了、看着、去过|
| 作宾语的人称代词：“我、你、他” | 找我、请你、麻烦他。 |
| 约定俗成 | 匀称、盘算、枇杷、篱笆、活泼、玄乎。狐狸、学生、拾掇、麻烦、蛤蟆、石榴。玫瑰、凉快、萝卜、朋友、奴才、云彩。脑袋、老爷、老婆、嘴巴、指头、指甲。委屈、喇叭、讲究、打发、打听、喜欢。点心、伙计、打扮、哑巴、女婿、首饰。自在、吓唬、力气、漂亮、队伍、地方。痛快、念叨、笑语、丈夫、志气、钥匙。月亮、正经、位置、秀气、上司、悟性。告示、动静、热闹、屁股、阔气、意思。等 |


### 2. "一" "不" 变调
#### "一" 变调
|  | 是否变调 | cases|
|:--|:-|:-|
| 单独念 | 否 | 第一、一楼|
| 序数 |否  | |
| 用在语句末尾 | 否 | |
| 去声前变阳平（四声前变二声） |  | 一栋yí dòng、一段yí duàn、一律yí lǜ、一路yí lù|
| 非去声前变去声（非四声前变四声） |  | 阴平（一声）<br>一发yì fā 、一端yì duān、一天yì tiān、一忽yì hū<br>阳平（二声）<br>一叠yì dié 、一同yì tóng 、一头yì tóu 、一条yì tiáo<br>上声（三声）<br>一统yì tǒng、一体yì tǐ、一览yì lǎn、一口yì kǒu|
|轻读，当“一”嵌在重叠式的动词之间  |  | 听一听 tīng yi tīng|

#### "不" 变调
|  | 是否变调 | cases|
|:--|:-|:-|
|单独念|否  | |
| 用在语句末尾| 否  | 我不|
|去声前变阳平（四声前变成二声）  |  | 不怕bú pà、不妙bú miào、不犯bú fàn、不忿bú fèn|
| 轻读，不”夹在重叠动词或重叠形容词之间、夹在动词和补语之间 |  |懂不懂 dǒng bu dǒng 、看不清 kàn bu qīng |


### 3. 三声变调
|  | 子类别| 如何变调|cases|
|:--|:-|:-|:-|
|单独念 |  | 否|  |
|句末 |  | 否|  |
|在句中停顿并没被后音节影响  |  |否 |  |
|三声+三声  |  | 二声+三声|保险、保养、党委、尽管、老板、本领、引导、古老、敏感、鼓舞、永远、语法、口语、岛屿、保姆、远景、北海、首长、母语 |
| 三个三声相连| 双音节+单音节（“双单格”结构）| 前两个变二声|演讲稿、跑马场、展览馆、管理组、水彩笔、蒙古语、选取法、古典舞、虎骨酒、洗脸水、草稿纸|
|  | 单音节+双音节（“单双格”结构）|第二个变二声|史小姐、党小组、好小伙、跑百米、纸老虎、李厂长、老保姆、冷处理、很友好、小雨伞|
|  | 单音节+单音节+单音节（“单三格”结构）| 前两个变二声| 软懒散、稳准狠|
| 更多三声音节相连时|  | 按语意与若干二字组成三字组，然后按以上变调规律处理|岂有 / 此理。<br>请你 / 给我 / 打点儿 / 洗脸水。<br>手表厂 / 有五种 /好产品。|

## References

 - [chinese_text_normalization](https://github.com/speechio/chinese_text_normalization)
 - [声调篇｜这些“一、不”变调规律，你不得不知](https://zhuanlan.zhihu.com/p/36156170)
 - [TTS前端模块中的普通话变调规则](https://zhuanlan.zhihu.com/p/65091429)
 - [轻声和变调](https://wenku.baidu.com/view/ad2016d94693daef5ef73db1.html)
 - [必读轻声词语表546条](http://www.chaziwang.com/article-view-504.html)
