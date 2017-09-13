download() {
    URL=$1
    MD5=$2
    TARGET=$3

    if [ -e $TARGET ]; then
        md5_result=`md5sum $TARGET | awk -F[' '] '{print $1}'`
        if [ $MD5 == $md5_result ]; then
            echo "$TARGET already exists, download skipped."
            return 0
        fi
    fi

    wget -c $URL -P `dirname "$TARGET"`
    md5_result=`md5sum $TARGET | awk -F[' '] '{print $1}'`
    if [ $MD5 -ne $md5_result ]; then
        echo "Fail to download the language model!"
        return 1
    fi
}
