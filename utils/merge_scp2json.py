#!/usr/bin/env python3
# encoding: utf-8
import argparse
import codecs
import json
import logging
import sys
from io import open

from paddlespeech.s2t.utils.cli_utils import get_commandline_args
from paddlespeech.utils.argparse import strtobool

PY2 = sys.version_info[0] == 2
sys.stdin = codecs.getreader("utf-8")(sys.stdin if PY2 else sys.stdin.buffer)
sys.stdout = codecs.getwriter("utf-8")(sys.stdout if PY2 else sys.stdout.buffer)


# Special types:
def shape(x):
    """Change str to List[int]

    >>> shape('3,5')
    [3, 5]
    >>> shape(' [3, 5] ')
    [3, 5]

    """

    # x: ' [3, 5] ' -> '3, 5'
    x = x.strip()
    if x[0] == "[":
        x = x[1:]
    if x[-1] == "]":
        x = x[:-1]

    return list(map(int, x.split(",")))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Given each file paths with such format as "
        "<key>:<file>:<type>. type> can be omitted and the default "
        'is "str". e.g. {} '
        "--input-scps feat:data/feats.scp shape:data/utt2feat_shape:shape "
        "--input-scps feat:data/feats2.scp shape:data/utt2feat2_shape:shape "
        "--output-scps text:data/text shape:data/utt2text_shape:shape "
        "--scps utt2spk:data/utt2spk".format(sys.argv[0]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument(
        "--input-scps",
        type=str,
        nargs="*",
        action="append",
        default=[],
        help="Json files for the inputs", )
    parser.add_argument(
        "--output-scps",
        type=str,
        nargs="*",
        action="append",
        default=[],
        help="Json files for the outputs", )
    parser.add_argument(
        "--scps",
        type=str,
        nargs="+",
        default=[],
        help="The json files except for the input and outputs", )
    parser.add_argument(
        "--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument(
        "--allow-one-column",
        type=strtobool,
        default=False,
        help="Allow one column in input scp files. "
        "In this case, the value will be empty string.", )
    parser.add_argument(
        "--out",
        "-O",
        type=str,
        help="The output filename. "
        "If omitted, then output to sys.stdout", )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.scps = [args.scps]

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    # List[List[Tuple[str, str, Callable[[str], Any], str, str]]]
    input_infos = []
    output_infos = []
    infos = []
    for lis_list, key_scps_list in [
        (input_infos, args.input_scps),
        (output_infos, args.output_scps),
        (infos, args.scps),
    ]:
        for key_scps in key_scps_list:
            lis = []
            for key_scp in key_scps:
                sps = key_scp.split(":")
                if len(sps) == 2:
                    key, scp = sps
                    type_func = None
                    type_func_str = "none"
                elif len(sps) == 3:
                    key, scp, type_func_str = sps
                    fail = False

                    try:
                        # type_func: Callable[[str], Any]
                        # e.g. type_func_str = "int" -> type_func = int
                        type_func = eval(type_func_str)
                    except Exception:
                        raise RuntimeError(
                            "Unknown type: {}".format(type_func_str))

                    if not callable(type_func):
                        raise RuntimeError(
                            "Unknown type: {}".format(type_func_str))

                else:
                    raise RuntimeError(
                        "Format <key>:<filepath> "
                        "or <key>:<filepath>:<type>  "
                        "e.g. feat:data/feat.scp "
                        "or shape:data/feat.scp:shape: {}".format(key_scp))

                for item in lis:
                    if key == item[0]:
                        raise RuntimeError('The key "{}" is duplicated: {} {}'.
                                           format(key, item[3], key_scp))

                lis.append((key, scp, type_func, key_scp, type_func_str))
            lis_list.append(lis)

    # Open  scp files
    input_fscps = [[open(i[1], "r", encoding="utf-8") for i in il]
                   for il in input_infos]
    output_fscps = [[open(i[1], "r", encoding="utf-8") for i in il]
                    for il in output_infos]
    fscps = [[open(i[1], "r", encoding="utf-8") for i in il] for il in infos]

    # Note(kamo): What is done here?
    # The final goal is creating a JSON file such as.
    # {
    #     "utts": {
    #         "sample_id1": {(omitted)},
    #         "sample_id2": {(omitted)},
    #          ....
    #     }
    # }
    #
    # To reduce memory usage, reading the input text files for each lines
    # and writing JSON elements per samples.
    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, "w", encoding="utf-8")
    out.write('{\n    "utts": {\n')
    nutt = 0
    while True:
        nutt += 1
        # List[List[str]]
        input_lines = [[f.readline() for f in fl] for fl in input_fscps]
        output_lines = [[f.readline() for f in fl] for fl in output_fscps]
        lines = [[f.readline() for f in fl] for fl in fscps]

        # Get the first line
        concat = sum(input_lines + output_lines + lines, [])
        if len(concat) == 0:
            break
        first = concat[0]

        # Sanity check: Must be sorted by the first column and have same keys
        count = 0
        for ls_list in (input_lines, output_lines, lines):
            for ls in ls_list:
                for line in ls:
                    if line == "" or first == "":
                        if line != first:
                            concat = sum(input_infos + output_infos + infos, [])
                            raise RuntimeError("The number of lines mismatch "
                                               'between: "{}" and "{}"'.format(
                                                   concat[0][1],
                                                   concat[count][1]))

                    elif line.split()[0] != first.split()[0]:
                        concat = sum(input_infos + output_infos + infos, [])
                        raise RuntimeError(
                            "The keys are mismatch at {}th line "
                            'between "{}" and "{}":\n>>> {}\n>>> {}'.format(
                                nutt,
                                concat[0][1],
                                concat[count][1],
                                first.rstrip(),
                                line.rstrip(), ))
                    count += 1

        # The end of file
        if first == "":
            if nutt != 1:
                out.write("\n")
            break
        if nutt != 1:
            out.write(",\n")

        entry = {}
        for inout, _lines, _infos in [
            ("input", input_lines, input_infos),
            ("output", output_lines, output_infos),
            ("other", lines, infos),
        ]:

            lis = []
            for idx, (line_list, info_list) in enumerate(
                    zip(_lines, _infos), 1):
                if inout == "input":
                    d = {"name": "input{}".format(idx)}
                elif inout == "output":
                    d = {"name": "target{}".format(idx)}
                else:
                    d = {}

                # info_list: List[Tuple[str, str, Callable]]
                # line_list: List[str]
                for line, info in zip(line_list, info_list):
                    sps = line.split(None, 1)
                    if len(sps) < 2:
                        if not args.allow_one_column:
                            raise RuntimeError(
                                "Format error {}th line in {}: "
                                ' Expecting "<key> <value>":\n>>> {}'.format(
                                    nutt, info[1], line))
                        uttid = sps[0]
                        value = ""
                    else:
                        uttid, value = sps

                    key = info[0]
                    type_func = info[2]
                    value = value.rstrip()

                    if type_func is not None:
                        try:
                            # type_func: Callable[[str], Any]
                            value = type_func(value)
                        except Exception:
                            logging.error(
                                '"{}" is an invalid function '
                                "for the {} th line in {}: \n>>> {}".format(
                                    info[4], nutt, info[1], line))
                            raise

                    d[key] = value
                lis.append(d)

            if inout != "other":
                entry[inout] = lis
            else:
                # If key == 'other'. only has the first item
                entry.update(lis[0])

        entry = json.dumps(
            entry,
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ": "))
        # Add indent
        indent = "    " * 2
        entry = ("\n" + indent).join(entry.split("\n"))

        uttid = first.split()[0]
        out.write('        "{}": {}'.format(uttid, entry))

    out.write("    }\n}\n")

    logging.info("{} entries in {}".format(nutt, out.name))
