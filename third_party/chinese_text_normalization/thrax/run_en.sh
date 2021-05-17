cd src
thraxmakedep en/verbalizer/podspeech.grm
make
cat ../testcase_en.txt
cat ../testcase_en.txt | thraxrewrite-tester --far=en/verbalizer/podspeech.far --rules=POD_SPEECH_TN
cd -
