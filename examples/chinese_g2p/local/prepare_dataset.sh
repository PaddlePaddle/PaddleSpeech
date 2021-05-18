echo "Extracting Prosody Labeling"

exp_dir="exp"
data_dir="data"
source ${MAIN_ROOT}/utils/parse_options.sh || exit -1

archive=${data_dir}/"BZNSYP.rar"
if [ ! -f ${archive} ]; then
    echo "Baker Dataset not found! Download it first to the data_dir."
    exit -1
fi

MD5='c4350563bf7dc298f7dd364b2607be83'
md5_result=$(md5sum ${archive} | awk -F[' '] '{print $1}')
if [ ${md5_result} != ${MD5} ]; then
    echo "MD5 mismatch! The Archive has been changed."
    exit -1
fi

   
label_file='ProsodyLabeling/000001-010000.txt'
filename='000001-010000.txt'
unrar e ${archive} ${label_file}
mv ${filename} ${exp_dir}

if [ ! -f ${exp_dir}/${filename} ];then
    echo "File extraction failed!"
    exit
fi

exit 0
