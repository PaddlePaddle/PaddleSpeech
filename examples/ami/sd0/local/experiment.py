# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import json
import os
import pickle
import shutil
import sys

import numpy as np
from tqdm.contrib import tqdm
from yacs.config import CfgNode

from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.cluster import diarization as diar
from utils.DER import DER

# Logger setup
logger = Log(__name__).getlog()


def diarize_dataset(
        full_meta,
        split_type,
        n_lambdas,
        pval,
        save_dir,
        config,
        n_neighbors=10, ):
    """This function diarizes all the recordings in a given dataset. It performs
    computation of embedding and clusters them using spectral clustering (or other backends).
    The output speaker boundary file is stored in the RTTM format.
    """

    # prepare `spkr_info` only once when Oracle num of speakers is selected.
    # spkr_info is essential to obtain number of speakers from groundtruth.
    if config.oracle_n_spkrs is True:
        full_ref_rttm_file = os.path.join(save_dir, config.ref_rttm_dir,
                                          "fullref_ami_" + split_type + ".rttm")
        rttm = diar.read_rttm(full_ref_rttm_file)

        spkr_info = list(  # noqa F841
            filter(lambda x: x.startswith("SPKR-INFO"), rttm))

    # get all the recording IDs in this dataset.
    all_keys = full_meta.keys()
    A = [word.rstrip().split("_")[0] for word in all_keys]
    all_rec_ids = list(set(A[1:]))
    all_rec_ids.sort()
    split = "AMI_" + split_type
    i = 1

    # adding tag for directory path.
    type_of_num_spkr = "oracle" if config.oracle_n_spkrs else "est"
    tag = (type_of_num_spkr + "_" + str(config.affinity) + "_" + config.backend)

    # make out rttm dir
    out_rttm_dir = os.path.join(save_dir, config.sys_rttm_dir, config.mic_type,
                                split, tag)
    if not os.path.exists(out_rttm_dir):
        os.makedirs(out_rttm_dir)

    # diarizing different recordings in a dataset.
    for rec_id in tqdm(all_rec_ids):
        # this tag will be displayed in the log.
        tag = ("[" + str(split_type) + ": " + str(i) + "/" +
               str(len(all_rec_ids)) + "]")
        i = i + 1

        # log message.
        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.debug(msg)

        # load embeddings.
        emb_file_name = rec_id + "." + config.mic_type + ".emb_stat.pkl"
        diary_stat_emb_file = os.path.join(save_dir, config.embedding_dir,
                                           split, emb_file_name)
        if not os.path.isfile(diary_stat_emb_file):
            msg = "Embdding file %s not found! Please check if embdding file is properly generated." % (
                diary_stat_emb_file)
            logger.error(msg)
            sys.exit()
        with open(diary_stat_emb_file, "rb") as in_file:
            diary_obj = pickle.load(in_file)

        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        # processing starts from here.
        if config.oracle_n_spkrs is True:
            # oracle num of speakers.
            num_spkrs = diar.get_oracle_num_spkrs(rec_id, spkr_info)
        else:
            if config.affinity == "nn":
                # num of speakers tunned on dev set (only for nn affinity).
                num_spkrs = n_lambdas
            else:
                # num of speakers will be estimated using max eigen gap for cos based affinity.
                # so adding None here. Will use this None later-on.
                num_spkrs = None

        if config.backend == "kmeans":
            diar.do_kmeans_clustering(
                diary_obj,
                out_rttm_file,
                rec_id,
                num_spkrs,
                pval, )

        if config.backend == "SC":
            # go for Spectral Clustering (SC).
            diar.do_spec_clustering(
                diary_obj,
                out_rttm_file,
                rec_id,
                num_spkrs,
                pval,
                config.affinity,
                n_neighbors, )

        # can used for AHC later. Likewise one can add different backends here.
        if config.backend == "AHC":
            # call AHC
            threshold = pval  # pval for AHC is nothing but threshold.
            diar.do_AHC(diary_obj, out_rttm_file, rec_id, num_spkrs, threshold)

    # once all RTTM outputs are generated, concatenate individual RTTM files to obtain single RTTM file.
    # this is not needed but just staying with the standards.
    concate_rttm_file = out_rttm_dir + "/sys_output.rttm"
    logger.debug("Concatenating individual RTTM files...")
    with open(concate_rttm_file, "w") as cat_file:
        for f in glob.glob(out_rttm_dir + "/*.rttm"):
            if f == concate_rttm_file:
                continue
            with open(f, "r") as indi_rttm_file:
                shutil.copyfileobj(indi_rttm_file, cat_file)

    msg = "The system generated RTTM file for %s set : %s" % (
        split_type, concate_rttm_file, )
    logger.debug(msg)

    return concate_rttm_file


def dev_pval_tuner(full_meta, save_dir, config):
    """Tuning p_value for affinity matrix.
    The p_value used so that only p% of the values in each row is retained.
    """

    DER_list = []
    prange = np.arange(0.002, 0.015, 0.001)

    n_lambdas = None  # using it as flag later.
    for p_v in prange:
        # Process whole dataset for value of p_v.
        concate_rttm_file = diarize_dataset(full_meta, "dev", n_lambdas, p_v,
                                            save_dir, config)

        ref_rttm_file = os.path.join(save_dir, config.ref_rttm_dir,
                                     "fullref_ami_dev.rttm")
        sys_rttm_file = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm_file,
            sys_rttm_file,
            config.ignore_overlap,
            config.forgiveness_collar, )

        DER_list.append(DER_)

        if config.oracle_n_spkrs is True and config.backend == "kmeans":
            # no need of p_val search. Note p_val is needed for SC for both oracle and est num of speakers.
            # p_val is needed in oracle_n_spkr=False when using kmeans backend.
            break

    # Take p_val that gave minmum DER on Dev dataset.
    tuned_p_val = prange[DER_list.index(min(DER_list))]

    return tuned_p_val


def dev_ahc_threshold_tuner(full_meta, save_dir, config):
    """Tuning threshold for affinity matrix. This function is called when AHC is used as backend.
    """

    DER_list = []
    prange = np.arange(0.0, 1.0, 0.1)

    n_lambdas = None  # using it as flag later.

    # Note: p_val is threshold in case of AHC.
    for p_v in prange:
        # Process whole dataset for value of p_v.
        concate_rttm_file = diarize_dataset(full_meta, "dev", n_lambdas, p_v,
                                            save_dir, config)

        ref_rttm = os.path.join(save_dir, config.ref_rttm_dir,
                                "fullref_ami_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            config.ignore_overlap,
            config.forgiveness_collar, )

        DER_list.append(DER_)

        if config.oracle_n_spkrs is True:
            break  # no need of threshold search.

    # Take p_val that gave minmum DER on Dev dataset.
    tuned_p_val = prange[DER_list.index(min(DER_list))]

    return tuned_p_val


def dev_nn_tuner(full_meta, split_type, save_dir, config):
    """Tuning n_neighbors on dev set. Assuming oracle num of speakers.
    This is used when nn based affinity is selected.
    """

    DER_list = []
    pval = None

    # Now assumming oracle num of speakers.
    n_lambdas = 4

    for nn in range(5, 15):

        # Process whole dataset for value of n_lambdas.
        concate_rttm_file = diarize_dataset(full_meta, "dev", n_lambdas, p_v,
                                            save_dir, config, nn)

        ref_rttm = os.path.join(save_dir, config.ref_rttm_dir,
                                "fullref_ami_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            config.ignore_overlap,
            config.forgiveness_collar, )

        DER_list.append([nn, DER_])

        if config.oracle_n_spkrs is True and config.backend == "kmeans":
            break

    DER_list.sort(key=lambda x: x[1])
    tunned_nn = DER_list[0]

    return tunned_nn[0]


def dev_tuner(full_meta, split_type, save_dir, config):
    """Tuning n_components on dev set. Used for nn based affinity matrix.
    Note: This is a very basic tunning for nn based affinity.
    This is work in progress till we find a better way.
    """

    DER_list = []
    pval = None
    for n_lambdas in range(1, config.max_num_spkrs + 1):

        # Process whole dataset for value of n_lambdas.
        concate_rttm_file = diarize_dataset(full_meta, "dev", n_lambdas, p_v,
                                            save_dir, config)

        ref_rttm = os.path.join(save_dir, config.ref_rttm_dir,
                                "fullref_ami_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            config.ignore_overlap,
            config.forgiveness_collar, )

        DER_list.append(DER_)

    # Take n_lambdas with minmum DER.
    tuned_n_lambdas = DER_list.index(min(DER_list)) + 1

    return tuned_n_lambdas


def main(args, config):
    # AMI Dev Set: Tune hyperparams on dev set.
    # Read the embdding file for dev set generated during embdding compute
    dev_meta_file = os.path.join(
        args.data_dir,
        config.meta_data_dir,
        "ami_dev." + config.mic_type + ".subsegs.json", )
    with open(dev_meta_file, "r") as f:
        meta_dev = json.load(f)

    full_meta = meta_dev

    # Processing starts from here
    # Following few lines selects option for different backend and affinity matrices. Finds best values for hyperameters using dev set.
    ref_rttm_file = os.path.join(args.data_dir, config.ref_rttm_dir,
                                 "fullref_ami_dev.rttm")
    best_nn = None
    if config.affinity == "nn":
        logger.info("Tuning for nn (Multiple iterations over AMI Dev set)")
        best_nn = dev_nn_tuner(full_meta, args.data_dir, config)

    n_lambdas = None
    best_pval = None

    if config.affinity == "cos" and (config.backend == "SC" or
                                     config.backend == "kmeans"):
        # oracle num_spkrs or not, doesn't matter for kmeans and SC backends
        # cos: Tune for the best pval for SC /kmeans (for unknown num of spkrs)
        logger.info(
            "Tuning for p-value for SC (Multiple iterations over AMI Dev set)")
        best_pval = dev_pval_tuner(full_meta, args.data_dir, config)

    elif config.backend == "AHC":
        logger.info("Tuning for threshold-value for AHC")
        best_threshold = dev_ahc_threshold_tuner(full_meta, args.data_dir,
                                                 config)
        best_pval = best_threshold
    else:
        # NN for unknown num of speakers (can be used in future)
        if config.oracle_n_spkrs is False:
            # nn: Tune num of number of components (to be updated later)
            logger.info(
                "Tuning for number of eigen components for NN (Multiple iterations over AMI Dev set)"
            )
            # dev_tuner used for tuning num of components in NN. Can be used in future.
            n_lambdas = dev_tuner(full_meta, args.data_dir, config)

    # load 'dev' and 'eval' metadata files.
    full_meta_dev = full_meta  # current full_meta is for 'dev'
    eval_meta_file = os.path.join(
        args.data_dir,
        config.meta_data_dir,
        "ami_eval." + config.mic_type + ".subsegs.json", )
    with open(eval_meta_file, "r") as f:
        full_meta_eval = json.load(f)

    # tag to be appended to final output DER files. Writing DER for individual files.
    type_of_num_spkr = "oracle" if config.oracle_n_spkrs else "est"
    tag = (
        type_of_num_spkr + "_" + str(config.affinity) + "." + config.mic_type)

    # perform final diarization on 'dev' and 'eval' with best hyperparams.
    final_DERs = {}
    out_der_dir = os.path.join(args.data_dir, config.der_dir)
    if not os.path.exists(out_der_dir):
        os.makedirs(out_der_dir)

    for split_type in ["dev", "eval"]:
        if split_type == "dev":
            full_meta = full_meta_dev
        else:
            full_meta = full_meta_eval

        # performing diarization.
        msg = "Diarizing using best hyperparams: " + split_type + " set"
        logger.info(msg)
        out_boundaries = diarize_dataset(
            full_meta,
            split_type,
            n_lambdas=n_lambdas,
            pval=best_pval,
            n_neighbors=best_nn,
            save_dir=args.data_dir,
            config=config)

        # computing DER.
        msg = "Computing DERs for " + split_type + " set"
        logger.info(msg)
        ref_rttm = os.path.join(args.data_dir, config.ref_rttm_dir,
                                "fullref_ami_" + split_type + ".rttm")
        sys_rttm = out_boundaries
        [MS, FA, SER, DER_vals] = DER(
            ref_rttm,
            sys_rttm,
            config.ignore_overlap,
            config.forgiveness_collar,
            individual_file_scores=True, )

        # writing DER values to a file. Append tag.
        der_file_name = split_type + "_DER_" + tag
        out_der_file = os.path.join(out_der_dir, der_file_name)
        msg = "Writing DER file to: " + out_der_file
        logger.info(msg)
        diar.write_ders_file(ref_rttm, DER_vals, out_der_file)

        msg = ("AMI " + split_type + " set DER = %s %%\n" %
               (str(round(DER_vals[-1], 2))))
        logger.info(msg)
        final_DERs[split_type] = round(DER_vals[-1], 2)

    # final print DERs
    msg = (
        "Final Diarization Error Rate (%%) on AMI corpus: Dev = %s %% | Eval = %s %%\n"
        % (str(final_DERs["dev"]), str(final_DERs["eval"])))
    logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--config", default=None, type=str, help="configuration file")
    parser.add_argument(
        "--data-dir",
        default="../data/",
        type=str,
        help="processsed data directory")
    args = parser.parse_args()
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    config.freeze()

    main(args, config)
