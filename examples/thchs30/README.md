this is the example of MFA for thchs30 dataset
cd a0 run run.sh to get start

MFA 对齐所使用的字典
MFA 字典的格式可以参考: https://montreal-forced-aligner.readthedocs.io/en/latest/dictionary.html
phone.lexicon 直接使用的是 THCHS-30/data_thchs30/lm_phone/lexicon.txt
word.lexicon 是一个带概率的字典, 生成规则请参考 local/gen_word2phone.py
