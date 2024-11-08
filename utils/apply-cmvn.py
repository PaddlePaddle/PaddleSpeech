#!/usr/bin/env python3
import argparse
import logging

import kaldiio
import numpy

from paddlespeech.audio.transform.cmvn import CMVN
from paddlespeech.s2t.utils.cli_readers import file_reader_helper
from paddlespeech.s2t.utils.cli_utils import get_commandline_args
from paddlespeech.s2t.utils.cli_utils import is_scipy_wav_style
from paddlespeech.s2t.utils.cli_writers import file_writer_helper
from paddlespeech.utils.argparse import strtobool


def get_parser():
    parser = argparse.ArgumentParser(
        description="apply mean-variance normalization to files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )

    parser.add_argument(
        "--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--in-filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi', )
    parser.add_argument(
        "--stats-filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5", "npy"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi', )
    parser.add_argument(
        "--out-filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5"],
        help="Specify the file format for the wspecifier. "
        '"mat" is the matrix format in kaldi', )

    parser.add_argument(
        "--norm-means",
        type=strtobool,
        default=True,
        help="Do variance normalization or not.", )
    parser.add_argument(
        "--norm-vars",
        type=strtobool,
        default=False,
        help="Do variance normalization or not.", )
    parser.add_argument(
        "--reverse",
        type=strtobool,
        default=False,
        help="Do reverse mode or not")
    parser.add_argument(
        "--spk2utt",
        type=str,
        help="A text file of speaker to utterance-list map. "
        "(Don't give rspecifier format, such as "
        '"ark:spk2utt")', )
    parser.add_argument(
        "--utt2spk",
        type=str,
        help="A text file of utterance to speaker map. "
        "(Don't give rspecifier format, such as "
        '"ark:utt2spk")', )
    parser.add_argument(
        "--write-num-frames",
        type=str,
        help="Specify wspecifer for utt2num_frames")
    parser.add_argument(
        "--compress",
        type=strtobool,
        default=False,
        help="Save in compressed format")
    parser.add_argument(
        "--compression-method",
        type=int,
        default=2,
        help="Specify the method(if mat) or "
        "gzip-level(if hdf5)", )
    parser.add_argument(
        "stats_rspecifier_or_rxfilename",
        help="Input stats. e.g. ark:stats.ark or stats.mat", )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier id. e.g. ark:some.ark")
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier id. e.g. ark:some.ark")
    return parser


def main():
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    if ":" in args.stats_rspecifier_or_rxfilename:
        is_rspcifier = True
        if args.stats_filetype == "npy":
            stats_filetype = "hdf5"
        else:
            stats_filetype = args.stats_filetype

        stats_dict = dict(
            file_reader_helper(args.stats_rspecifier_or_rxfilename,
                               stats_filetype))
    else:
        is_rspcifier = False
        if args.stats_filetype == "mat":
            stats = kaldiio.load_mat(args.stats_rspecifier_or_rxfilename)
        else:
            stats = numpy.load(args.stats_rspecifier_or_rxfilename)
        stats_dict = {None: stats}

    cmvn = CMVN(
        stats=stats_dict,
        norm_means=args.norm_means,
        norm_vars=args.norm_vars,
        utt2spk=args.utt2spk,
        spk2utt=args.spk2utt,
        reverse=args.reverse, )

    with file_writer_helper(
            args.wspecifier,
            filetype=args.out_filetype,
            write_num_frames=args.write_num_frames,
            compress=args.compress,
            compression_method=args.compression_method, ) as writer:
        for utt, mat in file_reader_helper(args.rspecifier, args.in_filetype):
            if is_scipy_wav_style(mat):
                # If data is sound file, then got as Tuple[int, ndarray]
                rate, mat = mat
            mat = cmvn(mat, utt if is_rspcifier else None)
            writer[utt] = mat


if __name__ == "__main__":
    main()
