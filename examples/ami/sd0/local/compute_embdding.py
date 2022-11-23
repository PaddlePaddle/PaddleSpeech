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
import json
import os
import pickle
import sys

import numpy as np
import paddle
from paddle.io import BatchSampler
from paddle.io import DataLoader
from tqdm.contrib import tqdm
from yacs.config import CfgNode

from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.cluster.diarization import EmbeddingMeta
from paddlespeech.vector.io.batch import batch_feature_normalize
from paddlespeech.vector.io.dataset_from_json import JSONDataset
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.modules.sid_model import SpeakerIdetification
from paddlespeech.vector.training.seeding import seed_everything

# Logger setup
logger = Log(__name__).getlog()


def prepare_subset_json(full_meta_data, rec_id, out_meta_file):
    """Prepares metadata for a given recording ID.

    Arguments
    ---------
    full_meta_data : json
        Full meta (json) containing all the recordings
    rec_id : str
        The recording ID for which meta (json) has to be prepared
    out_meta_file : str
        Path of the output meta (json) file.
    """

    subset = {}
    for key in full_meta_data:
        k = str(key)
        if k.startswith(rec_id):
            subset[key] = full_meta_data[key]

    with open(out_meta_file, mode="w") as json_f:
        json.dump(subset, json_f, indent=2)


def create_dataloader(json_file, batch_size):
    """Creates the datasets and their data processing pipelines.
    This is used for multi-mic processing.
    """

    # create datasets
    dataset = JSONDataset(
        json_file=json_file,
        feat_type='melspectrogram',
        n_mels=config.n_mels,
        window_size=config.window_size,
        hop_length=config.hop_size)

    # create dataloader
    batch_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            collate_fn=lambda x: batch_feature_normalize(
                                x, mean_norm=True, std_norm=False),
                            return_list=True)

    return dataloader


def main(args, config):
    # set the training device, cpu or gpu
    paddle.set_device(args.device)
    # set the random seed
    seed_everything(config.seed)

    # stage1: build the dnn backbone model network
    ecapa_tdnn = EcapaTdnn(**config.model)

    # stage2: build the speaker verification eval instance with backbone model
    model = SpeakerIdetification(backbone=ecapa_tdnn, num_class=1)

    # stage3: load the pre-trained model
    #         we get the last model from the epoch and save_interval
    args.load_checkpoint = os.path.abspath(
        os.path.expanduser(args.load_checkpoint))

    # load model checkpoint to sid model
    state_dict = paddle.load(
        os.path.join(args.load_checkpoint, 'model.pdparams'))
    model.set_state_dict(state_dict)
    logger.info(f'Checkpoint loaded from {args.load_checkpoint}')

    # set the model to eval mode
    model.eval()

    # load meta data
    meta_file = os.path.join(
        args.data_dir,
        config.meta_data_dir,
        "ami_" + args.dataset + "." + config.mic_type + ".subsegs.json", )
    with open(meta_file, "r") as f:
        full_meta = json.load(f)

    # get all the recording IDs in this dataset.
    all_keys = full_meta.keys()
    A = [word.rstrip().split("_")[0] for word in all_keys]
    all_rec_ids = list(set(A[1:]))
    all_rec_ids.sort()
    split = "AMI_" + args.dataset
    i = 1

    msg = "Extra embdding for " + args.dataset + " set"
    logger.info(msg)

    if len(all_rec_ids) <= 0:
        msg = "No recording IDs found! Please check if meta_data json file is properly generated."
        logger.error(msg)
        sys.exit()

    # extra different recordings embdding in a dataset.
    for rec_id in tqdm(all_rec_ids):
        # This tag will be displayed in the log.
        tag = ("[" + str(args.dataset) + ": " + str(i) + "/" +
               str(len(all_rec_ids)) + "]")
        i = i + 1

        # log message.
        msg = "Embdding %s : %s " % (tag, rec_id)
        logger.debug(msg)

        # embedding directory.
        if not os.path.exists(
                os.path.join(args.data_dir, config.embedding_dir, split)):
            os.makedirs(
                os.path.join(args.data_dir, config.embedding_dir, split))

        # file to store embeddings.
        emb_file_name = rec_id + "." + config.mic_type + ".emb_stat.pkl"
        diary_stat_emb_file = os.path.join(args.data_dir, config.embedding_dir,
                                           split, emb_file_name)

        # prepare a metadata (json) for one recording. This is basically a subset of full_meta.
        # lets keep this meta-info in embedding directory itself.
        json_file_name = rec_id + "." + config.mic_type + ".json"
        meta_per_rec_file = os.path.join(args.data_dir, config.embedding_dir,
                                         split, json_file_name)

        # write subset (meta for one recording) json metadata.
        prepare_subset_json(full_meta, rec_id, meta_per_rec_file)

        # prepare data loader.
        diary_set_loader = create_dataloader(meta_per_rec_file,
                                             config.batch_size)

        # extract embeddings (skip if already done).
        if not os.path.isfile(diary_stat_emb_file):
            logger.debug("Extracting deep embeddings")
            embeddings = np.empty(shape=[0, config.emb_dim], dtype=np.float64)
            segset = []

            for batch_idx, batch in enumerate(tqdm(diary_set_loader)):
                # extrac the audio embedding
                ids, feats, lengths = batch['ids'], batch['feats'], batch[
                    'lengths']
                seg = [x for x in ids]
                segset = segset + seg
                emb = model.backbone(feats, lengths).squeeze(
                    -1).numpy()  # (N, emb_size, 1) -> (N, emb_size)
                embeddings = np.concatenate((embeddings, emb), axis=0)

            segset = np.array(segset, dtype="|O")
            stat_obj = EmbeddingMeta(
                segset=segset,
                stats=embeddings, )
            logger.debug("Saving Embeddings...")
            with open(diary_stat_emb_file, "wb") as output:
                pickle.dump(stat_obj, output)

        else:
            logger.debug("Skipping embedding extraction (as already present).")


# Begin experiment!
if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        '--device',
        default="gpu",
        help="Select which device to perform diarization, defaults to gpu.")
    parser.add_argument(
        "--config", default=None, type=str, help="configuration file")
    parser.add_argument(
        "--data-dir",
        default="../save/",
        type=str,
        help="processsed data directory")
    parser.add_argument(
        "--dataset",
        choices=['dev', 'eval'],
        default="dev",
        type=str,
        help="Select which dataset to extra embdding, defaults to dev")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default='',
        help="Directory to load model checkpoint to compute embeddings.")
    args = parser.parse_args()
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    config.freeze()

    main(args, config)
