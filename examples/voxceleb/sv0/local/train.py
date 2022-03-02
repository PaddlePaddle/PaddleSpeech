import argparse
import paddle
from dataset.voxceleb.voxceleb1 import VoxCeleb1


def main(args):
    paddle.set_device(args.device)

    # stage1: we must call the paddle.distributed.init_parallel_env() api at the begining
    paddle.distributed.init_parallel_env()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()

    # stage2: data prepare
    train_ds = VoxCeleb1('train', target_dir=args.data_dir)

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', 
                        choices=['cpu', 'gpu'], 
                        default="cpu", 
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--data-dir",
                        default="./data/",
                        type=str,
                        help="data directory")
    args = parser.parse_args()
    # yapf: enable

    main(args)