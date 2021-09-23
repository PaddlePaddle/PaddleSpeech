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

    wget -c $URL -O "$TARGET"
    if [ $? -ne 0 ]; then
        return 1
    fi

    md5_result=`md5sum $TARGET | awk -F[' '] '{print $1}'`
    if [ ! $MD5 == $md5_result ]; then
        return 1
    fi
}
