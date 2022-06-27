# copy this to root directory of data and 
# chmod a+x convert.sh
# ./convert.sh
# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
dir=$1
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}
N=32 # number of vCPU
open_sem $N
for f in $(find ${dir} -name "*.m4a"); do
    run_with_lock ffmpeg -loglevel panic -i "$f" -ar 16000 "${f%.*}.wav"
done
