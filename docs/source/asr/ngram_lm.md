# Ngram LM

## Prepare Language Model

A language model is required to improve the decoder's performance. We have prepared two language models (with lossy compression) for users to download and try. One is for English and the other is for Mandarin. The bash script to download LM is example's `local/download_lm_*.sh`.

For example, users can simply run this to download the prepared mandarin language models:

```bash
cd examples/aishell
source path.sh
bash local/download_lm_ch.sh
```
If you wish to train your own better language model, please refer to [KenLM](https://github.com/kpu/kenlm) for tutorials.
Here we provide some tips to show how we prepare our English and Mandarin language models.
You can take it as a reference when you train your own.

### English LM

The English corpus is from the [Common Crawl Repository](http://commoncrawl.org) and you can download it from [statmt](http://data.statmt.org/ngrams/deduped_en). We use part en.00 to train our English language model. There are some preprocessing steps before training:

  * Characters not in \['A-Za-z0-9\s'\] (\s represents whitespace characters) are removed and Arabic numbers are converted to English numbers like 1000 to one thousand.
  * Repeated whitespace characters are squeezed to one and the beginning whitespace characters are removed. Notice that all transcriptions are lowercase, so all characters are converted to lowercase.
  * Top 400,000 most frequent words are selected to build the vocabulary and the rest are replaced with 'UNKNOWNWORD'.

Now the preprocessing is done and we get a clean corpus to train the language model. Our released language model is trained with arguments '-o 5 --prune 0 1 1 1 1'. '-o 5' means the max order of the language model is 5. '--prune 0 1 1 1 1' represents count thresholds for each order and more specifically it will prune singletons for orders two and higher. To save disk storage we convert the ARPA file to 'trie' binary file with arguments '-a 22 -q 8 -b 8'. '-a' represents the maximum number of leading bits of pointers in 'trie' to chop. '-q -b' are quantization parameters for probability and backoff.

### Mandarin LM

Different from the English language model, the Mandarin language model is character-based where each token is a Chinese character. We use the internal corpus to train the released Mandarin language models. The corpus contains billions of tokens. The preprocessing has a tiny difference from the English language model and the main steps include:

  * The beginning and trailing whitespace characters are removed.
  * English punctuations and Chinese punctuations are removed.
  * A whitespace character between two tokens is inserted.

Please notice that the released language models only contain Chinese simplified characters. After preprocessing is done we can begin to train the language model. The key training arguments for small LM are '-o 5 --prune 0 1 2 4 4' and '-o 5' for large LM. Please refer above section for the meaning of each argument. We also convert the ARPA file to a binary file using default settings.
