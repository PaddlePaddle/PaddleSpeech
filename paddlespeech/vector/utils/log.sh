_HDR_FMT="%.23s %s[%s]: "
_ERR_MSG_FMT="ERROR: ${_HDR_FMT}%s\n"
_INFO_MSG_FMT="INFO: ${_HDR_FMT}%s\n"

error_msg() {
  printf "$_ERR_MSG_FMT" $(date +%F.%T.%N) ${BASH_SOURCE[1]##*/} ${BASH_LINENO[0]} "${@}"
}

info_msg() {
  printf "$_INFO_MSG_FMT" $(date +%F.%T.%N) ${BASH_SOURCE[1]##*/} ${BASH_LINENO[0]} "${@}"
}
