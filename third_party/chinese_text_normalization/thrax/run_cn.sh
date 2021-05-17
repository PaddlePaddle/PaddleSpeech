cd src/cn
thraxmakedep itn.grm
make
#thraxrewrite-tester --far=itn.far --rules=ITN 
cat ../../testcase_cn.txt | thraxrewrite-tester --far=itn.far --rules=ITN 
cd -
