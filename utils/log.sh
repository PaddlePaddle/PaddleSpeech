_HDR_FMT="%.23s %s[%s]: "
_ERR_MSG_FMT="ERROR: ${_HDR_FMT}%s\n"
_INFO_MSG_FMT="INFO: ${_HDR_FMT}%s\n"

script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x $script_dir/../paddle_log
$script_dir/../paddle_log

error_msg() {
  printf "$_ERR_MSG_FMT" $(date +%F.%T.%N) ${BASH_SOURCE[1]##*/} ${BASH_LINENO[0]} "${@}"
}

info_msg() {
  printf "$_INFO_MSG_FMT" $(date +%F.%T.%N) ${BASH_SOURCE[1]##*/} ${BASH_LINENO[0]} "${@}"
}
