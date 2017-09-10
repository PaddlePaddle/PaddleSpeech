echo "Downloading language model ..."

mkdir data

LM=common_crawl_00.prune01111.trie.klm
MD5="099a601759d467cd0a8523ff939819c5"

wget -c http://paddlepaddle.bj.bcebos.com/model_zoo/speech/$LM -P ./data

echo "Checking md5sum ..."
md5_tmp=`md5sum ./data/$LM | awk -F[' '] '{print $1}'`

if [ $MD5 != $md5_tmp ]; then
    echo "Fail to download the language model!"
    exit 1
fi
