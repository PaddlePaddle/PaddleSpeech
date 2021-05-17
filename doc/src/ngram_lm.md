# Ngram LM

## Prepare Language Model

A language model is required to improve the decoder's performance. We have prepared two language models (with lossy compression) for users to download and try. One is for English and the other is for Mandarin. The bash script to download LM is example's `local/download_lm_*.sh`.

For example, users can simply run this to download the preprared mandarin language models:

```bash
cd examples/aishell
source path.sh
bash local/download_lm_ch.sh
```

If you wish to train your own better language model, please refer to [KenLM](https://github.com/kpu/kenlm) for tutorials.
Here we provide some tips to show how we preparing our English and Mandarin language models.
You can take it as a reference when you train your own.

### English LM

The English corpus is from the [Common Crawl Repository](http://commoncrawl.org) and you can download it from [statmt](http://data.statmt.org/ngrams/deduped_en). We use part en.00 to train our English language model. There are some preprocessing steps before training:

  * Characters not in \['A-Za-z0-9\s'\] (\s represents whitespace characters) are removed and Arabic numbers are converted to English numbers like 1000 to one thousand.
  * Repeated whitespace characters are squeezed to one and the beginning whitespace characters are removed. Notice that all transcriptions are lowercase, so all characters are converted to lowercase.
  * Top 400,000 most frequent words are selected to build the vocabulary and the rest are replaced with 'UNKNOWNWORD'.

Now the preprocessing is done and we get a clean corpus to train the language model. Our released language model are trained with agruments '-o 5 --prune 0 1 1 1 1'. '-o 5' means the max order of language model is 5. '--prune 0 1 1 1 1' represents count thresholds for each order and more specifically it will prune singletons for orders two and higher. To save disk storage we convert the arpa file to 'trie' binary file with arguments '-a 22 -q 8 -b 8'. '-a' represents the maximum number of leading bits of pointers in 'trie' to chop. '-q -b' are quantization parameters for probability and backoff.

### Mandarin LM

Different from the English language model, Mandarin language model is character-based where each token is a Chinese character. We use internal corpus to train the released Mandarin language models. The corpus contain billions of tokens. The preprocessing has tiny difference from English language model and main steps include:

  * The beginning and trailing whitespace characters are removed.
  * English punctuations and Chinese punctuations are removed.
  * A whitespace character between two tokens is inserted.

Please notice that the released language models only contain Chinese simplified characters. After preprocessing done we can begin to train the language model. The key training arguments for small LM is '-o 5 --prune 0 1 2 4 4' and '-o 5' for large LM. Please refer above section for the meaning of each argument. We also convert the arpa file to binary file using default settings.



## [KenLM](http://kheafield.com/code/kenlm/)

统计语言模型工具有比较多的选择，目前使用比较好的有srilm及kenlm，其中kenlm比srilm晚出来，训练速度也更快，而且支持单机大数据的训练。现在介绍一下kenlm的使用方法。

1. 工具包的下载地址：http://kheafield.com/code/kenlm.tar.gz

2. 使用。该工具在linux环境下使用方便。 先确保linux环境已经按照1.36.0的Boost和zlib

   ```
   boost:
   yum install boost
   yum install boost-devel

   zlib:
   yum install zlib
   yum install zlib-devel
   ```

   然后gcc版本需要是4.8.2及以上。

   ```
   wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
   mkdir kenlm/build
   cd kenlm/build
   cmake ..
   make -j2
   ```

3. 训练。使用如下命令进行训练：

   ```
   build/bin/lmplz -o 3 --verbose_header --text people2014corpus_words.txt --arpa result/people2014corpus_words.arps
   ```

   其中，
   1）people2014corpus_words.txt文件必须是分词以后的文件。

   训练语料<人民日报2014版熟语料>，包括： 1）标准人工切词及词性数据people2014.tar.gz， 2）未切词文本数据people2014_words.txt， 3）kenlm训练字粒度语言模型文件及其二进制文件people2014corpus_chars.arps/klm， 4）kenlm词粒度语言模型文件及其二进制文件people2014corpus_words.arps/klm。

   2）-o后面的5表示的是5-gram,一般取到3即可，但可以结合自己实际情况判断。

4. 压缩。压缩模型为二进制，方便模型快速加载：

   ```
   build/bin/build_binary ./result/people2014corpus_words.arps ./result/people2014corpus_words.klm
   ```