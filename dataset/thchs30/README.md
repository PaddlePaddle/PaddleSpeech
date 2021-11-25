# [THCHS30](http://www.openslr.org/18/)

This is the *data part* of the `THCHS30 2015` acoustic data
& scripts dataset.

The dataset is described in more detail in the paper ``THCHS-30 : A Free
Chinese Speech Corpus`` by Dong Wang, Xuewei Zhang.

A paper (if it can be called a paper) 13 years ago regarding the database:

Dong Wang, Dalei Wu, Xiaoyan Zhu, ``TCMSD: A new Chinese Continuous Speech Database``,
International Conference on Chinese Computing (ICCC'01), 2001, Singapore.

The layout of this data pack is the following:

  ``data``
      ``*.wav``
        audio data

      ``*.wav.trn``  
        transcriptions

  ``{train,dev,test}``
    contain symlinks into the ``data`` directory for both audio and
    transcription files. Contents of these directories define the
    train/dev/test split of the data.

  ``{lm_word}``
       ``word.3gram.lm``
         trigram LM based on word
        ``lexicon.txt``
         lexicon based on word

   ``{lm_phone}``
       ``phone.3gram.lm``
         trigram LM based on phone
        ``lexicon.txt``
         lexicon based on phone

  ``README.TXT``
    this file


Data statistics
===============

Statistics for the data are as follows:

    ===========  ==========  ==========  ===========
    **dataset**  **audio**   **#sents**  **#words**
    ===========  ==========  ==========  ===========
        train        25        10,000      198,252
        dev         2:14         893        17,743
        test        6:15        2,495       49,085
    ===========  ==========  ==========  ===========
