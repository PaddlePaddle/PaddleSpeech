#!/usr/bin/env python3
import argparse
import logging
import sys

from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.cli_readers import file_reader_helper
from paddlespeech.s2t.utils.cli_utils import get_commandline_args
from paddlespeech.s2t.utils.cli_utils import is_scipy_wav_style


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert feature to its shape",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument(
        "--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi', )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing", )
    parser.add_argument(
        "rspecifier",
        type=str,
        help="Read specifier for feats. e.g. ark:some.ark")
    parser.add_argument(
        "out",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The output filename. "
        "If omitted, then output to sys.stdout", )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    if args.preprocess_conf is not None:
        preprocessing = Transformation(args.preprocess_conf)
        logging.info("Apply preprocessing: {}".format(preprocessing))
    else:
        preprocessing = None

    # There are no necessary for matrix without preprocessing,
    # so change to file_reader_helper to return shape.
    # This make sense only with filetype="hdf5".
    for utt, mat in file_reader_helper(
            args.rspecifier, args.filetype, return_shape=preprocessing is None):
        if preprocessing is not None:
            if is_scipy_wav_style(mat):
                # If data is sound file, then got as Tuple[int, ndarray]
                rate, mat = mat
            mat = preprocessing(mat, uttid_list=utt)
            shape_str = ",".join(map(str, mat.shape))
        else:
            if len(mat) == 2 and isinstance(mat[1], tuple):
                # If data is sound file, Tuple[int, Tuple[int, ...]]
                rate, mat = mat
            shape_str = ",".join(map(str, mat))
        args.out.write("{} {}\n".format(utt, shape_str))


if __name__ == "__main__":
    main()
