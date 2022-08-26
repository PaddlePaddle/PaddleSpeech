set -ex
NNODES=${PADDLE_TRAINERS_NUM:-"1"}
PYTHON=${PYTHON:-"python"}
TIMEOUT=${1:-"10m"}

if [[ "$NNODES" -gt 1 ]]; then
  while ! timeout "$TIMEOUT" "$PYTHON" -m paddle.distributed.launch run_check; do
    echo "Retry barrier ......"
  done
fi
