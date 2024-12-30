#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging
import sys

from paddlespeech.utils.argparse import strtobool

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description="add multiple json values to an input or output value",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("jsons", type=str, nargs="+", help="json files")
    parser.add_argument(
        "-i",
        "--is-input",
        default=True,
        type=strtobool,
        help="If true, add to input. If false, add to output", )
    parser.add_argument(
        "--verbose", "-V", default=0, type=int, help="Verbose option")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(args)

    # make intersection set for utterance keys
    js = []
    intersec_ks = []
    for x in args.jsons:
        with codecs.open(x, "r", encoding="utf-8") as f:
            j = json.load(f)
        ks = j["utts"].keys()
        logging.info(x + ": has " + str(len(ks)) + " utterances")
        if len(intersec_ks) > 0:
            intersec_ks = intersec_ks.intersection(set(ks))
            if len(intersec_ks) == 0:
                logging.warning("Empty intersection")
                break
        else:
            intersec_ks = set(ks)
        js.append(j)
    logging.info("new json has " + str(len(intersec_ks)) + " utterances")

    # updated original dict to keep intersection
    intersec_org_dic = dict()
    for k in intersec_ks:
        v = js[0]["utts"][k]
        intersec_org_dic[k] = v

    intersec_add_dic = dict()
    for k in intersec_ks:
        v = js[1]["utts"][k]
        for j in js[2:]:
            v.update(j["utts"][k])
        intersec_add_dic[k] = v

    new_dic = dict()
    for key_id in intersec_org_dic:
        orgdic = intersec_org_dic[key_id]
        adddic = intersec_add_dic[key_id]

        if "utt2spk" not in orgdic:
            orgdic["utt2spk"] = ""
        # NOTE: for machine translation

        # add as input
        if args.is_input:
            # original input
            input_list = orgdic["input"]
            # additional input
            in_add_dic = {}
            if "idim" in adddic and "ilen" in adddic:
                in_add_dic["shape"] = [int(adddic["ilen"]), int(adddic["idim"])]
            elif "idim" in adddic:
                in_add_dic["shape"] = [int(adddic["idim"])]
            # add all other key value
            for key, value in adddic.items():
                if key in ["idim", "ilen"]:
                    continue
                in_add_dic[key] = value
            # add name
            in_add_dic["name"] = "input%d" % (len(input_list) + 1)

            input_list.append(in_add_dic)
            new_dic[key_id] = {
                "input": input_list,
                "output": orgdic["output"],
                "utt2spk": orgdic["utt2spk"],
            }
        # add as output
        else:
            # original output
            output_list = orgdic["output"]
            # additional output
            out_add_dic = {}
            # add shape
            if "odim" in adddic and "olen" in adddic:
                out_add_dic[
                    "shape"] = [int(adddic["olen"]), int(adddic["odim"])]
            elif "odim" in adddic:
                out_add_dic["shape"] = [int(adddic["odim"])]
            # add all other key value
            for key, value in adddic.items():
                if key in ["odim", "olen"]:
                    continue
                out_add_dic[key] = value
            # add name
            out_add_dic["name"] = "target%d" % (len(output_list) + 1)

            output_list.append(out_add_dic)
            new_dic[key_id] = {
                "input": orgdic["input"],
                "output": output_list,
                "utt2spk": orgdic["utt2spk"],
            }
            if "lang" in orgdic.keys():
                new_dic[key_id]["lang"] = orgdic["lang"]

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(
        {
            "utts": new_dic
        },
        indent=4,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ": "), )
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout
                                           if is_python2 else sys.stdout.buffer)
    print(jsonstring)
