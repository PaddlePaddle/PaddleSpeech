# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os

import numpy as np
import paddle
from paddle.io import BatchSampler
from paddle.io import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from paddleaudio.datasets import VoxCeleb
from paddleaudio.metric import compute_eer
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.io.batch import batch_feature_normalize
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.modules.sid_model import SpeakerIdetification
from paddlespeech.vector.training.seeding import seed_everything

logger = Log(__name__).getlog()


def main(args, config):
    # stage0: set the training device, cpu or gpu
    paddle.set_device(args.device)
    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)

    # stage1: build the dnn backbone model network
    ecapa_tdnn = EcapaTdnn(**config.model)

    # stage2: build the speaker verification eval instance with backbone model
    model = SpeakerIdetification(
        backbone=ecapa_tdnn, num_class=config.num_speakers)

    # stage3: load the pre-trained model
    #         we get the last model from the epoch and save_interval
    args.load_checkpoint = os.path.abspath(
        os.path.expanduser(args.load_checkpoint))

    # load model checkpoint to sid model
    state_dict = paddle.load(
        os.path.join(args.load_checkpoint, 'model.pdparams'))
    model.set_state_dict(state_dict)
    logger.info(f'Checkpoint loaded from {args.load_checkpoint}')

    # stage4: construct the enroll and test dataloader

    enroll_dataset = VoxCeleb(
        subset='enroll',
        target_dir=args.data_dir,
        feat_type='melspectrogram',
        random_chunk=False,
        n_mels=config.n_mels,
        window_size=config.window_size,
        hop_length=config.hop_size)
    enroll_sampler = BatchSampler(
        enroll_dataset, batch_size=config.batch_size,
        shuffle=True)  # Shuffle to make embedding normalization more robust.
    enrol_loader = DataLoader(enroll_dataset,
                    batch_sampler=enroll_sampler,
                    collate_fn=lambda x: batch_feature_normalize(
                            x, mean_norm=True, std_norm=False),
                    num_workers=config.num_workers,
                    return_list=True,)
    test_dataset = VoxCeleb(
        subset='test',
        target_dir=args.data_dir,
        feat_type='melspectrogram',
        random_chunk=False,
        n_mels=config.n_mels,
        window_size=config.window_size,
        hop_length=config.hop_size)

    test_sampler = BatchSampler(
        test_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,
                            batch_sampler=test_sampler,
                            collate_fn=lambda x: batch_feature_normalize(
                                x, mean_norm=True, std_norm=False),
                            num_workers=config.num_workers,
                            return_list=True,)
    # stage5: we must set the model to eval mode
    model.eval()

    # stage6: global embedding norm to imporve the performance
    logger.info(f"global embedding norm: {config.global_embedding_norm}")
    if config.global_embedding_norm:
        global_embedding_mean = None
        global_embedding_std = None
        mean_norm_flag = config.embedding_mean_norm
        std_norm_flag = config.embedding_std_norm
        batch_count = 0

    # stage7: Compute embeddings of audios in enrol and test dataset from model.
    id2embedding = {}
    # Run multi times to make embedding normalization more stable.
    for i in range(2):
        for dl in [enrol_loader, test_loader]:
            logger.info(
                f'Loop {[i+1]}: Computing embeddings on {dl.dataset.subset} dataset'
            )
            with paddle.no_grad():
                for batch_idx, batch in enumerate(tqdm(dl)):

                    # stage 8-1: extrac the audio embedding
                    ids, feats, lengths = batch['ids'], batch['feats'], batch[
                        'lengths']
                    embeddings = model.backbone(feats, lengths).squeeze(
                        -1).numpy()  # (N, emb_size, 1) -> (N, emb_size)

                    # Global embedding normalization.
                    # if we use the global embedding norm
                    # eer can reduece about relative 10%
                    if config.global_embedding_norm:
                        batch_count += 1
                        current_mean = embeddings.mean(
                            axis=0) if mean_norm_flag else 0
                        current_std = embeddings.std(
                            axis=0) if std_norm_flag else 1
                        # Update global mean and std.
                        if global_embedding_mean is None and global_embedding_std is None:
                            global_embedding_mean, global_embedding_std = current_mean, current_std
                        else:
                            weight = 1 / batch_count  # Weight decay by batches.
                            global_embedding_mean = (
                                1 - weight
                            ) * global_embedding_mean + weight * current_mean
                            global_embedding_std = (
                                1 - weight
                            ) * global_embedding_std + weight * current_std
                        # Apply global embedding normalization.
                        embeddings = (embeddings - global_embedding_mean
                                      ) / global_embedding_std

                    # Update embedding dict.
                    id2embedding.update(dict(zip(ids, embeddings)))

    # stage 8: Compute cosine scores.
    labels = []
    enroll_ids = []
    test_ids = []
    logger.info(f"read the trial from {VoxCeleb.veri_test_file}")
    with open(VoxCeleb.veri_test_file, 'r') as f:
        for line in f.readlines():
            label, enroll_id, test_id = line.strip().split(' ')
            labels.append(int(label))
            enroll_ids.append(enroll_id.split('.')[0].replace('/', '-'))
            test_ids.append(test_id.split('.')[0].replace('/', '-'))

    cos_sim_func = paddle.nn.CosineSimilarity(axis=1)
    enrol_embeddings, test_embeddings = map(lambda ids: paddle.to_tensor(
        np.asarray([id2embedding[uttid] for uttid in ids], dtype='float32')),
                                            [enroll_ids, test_ids
                                             ])  # (N, emb_size)
    scores = cos_sim_func(enrol_embeddings, test_embeddings)
    EER, threshold = compute_eer(np.asarray(labels), scores.numpy())
    logger.info(
        f'EER of verification test: {EER*100:.4f}%, score threshold: {threshold:.5f}'
    )


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device',
                        choices=['cpu', 'gpu'],
                        default="gpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--config",
                        default=None,
                        type=str,
                        help="configuration file")
    parser.add_argument("--data-dir",
                        default="./data/",
                        type=str,
                        help="data directory")
    parser.add_argument("--load-checkpoint",
                        type=str,
                        default='',
                        help="Directory to load model checkpoint to contiune trainning.")
    args = parser.parse_args()
    # yapf: enable
    # https://yaml.org/type/float.html
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    config.freeze()
    print(config)
    main(args, config)
