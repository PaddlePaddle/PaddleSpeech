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

from paddleaudio.metric import compute_eer
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.io.batch import batch_feature_normalize
from paddlespeech.vector.io.dataset import CSVDataset
from paddlespeech.vector.io.embedding_norm import InputNormalization
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.modules.sid_model import SpeakerIdetification
from paddlespeech.vector.training.seeding import seed_everything

logger = Log(__name__).getlog()


def compute_dataset_embedding(data_loader, model, mean_var_norm_emb, config,
                              id2embedding):
    """compute the dataset embeddings

    Args:
        data_loader (_type_): _description_
        model (_type_): _description_
        mean_var_norm_emb (_type_): _description_
        config (_type_): _description_
    """
    logger.info(
        f'Computing embeddings on {data_loader.dataset.csv_path} dataset')
    with paddle.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):

            # stage 8-1: extrac the audio embedding
            ids, feats, lengths = batch['ids'], batch['feats'], batch['lengths']
            embeddings = model.backbone(feats, lengths).squeeze(
                -1)  # (N, emb_size, 1) -> (N, emb_size)

            # Global embedding normalization.
            # if we use the global embedding norm
            # eer can reduece about relative 10%
            if config.global_embedding_norm and mean_var_norm_emb:
                lengths = paddle.ones([embeddings.shape[0]])
                embeddings = mean_var_norm_emb(embeddings, lengths)

            # Update embedding dict.
            id2embedding.update(dict(zip(ids, embeddings)))


def compute_verification_scores(id2embedding, train_cohort, config):
    labels = []
    enroll_ids = []
    test_ids = []
    logger.info(f"read the trial from {config.verification_file}")
    cos_sim_func = paddle.nn.CosineSimilarity(axis=-1)
    scores = []
    with open(config.verification_file, 'r') as f:
        for line in f.readlines():
            label, enroll_id, test_id = line.strip().split(' ')
            enroll_id = enroll_id.split('.')[0].replace('/', '-')
            test_id = test_id.split('.')[0].replace('/', '-')
            labels.append(int(label))

            enroll_emb = id2embedding[enroll_id]
            test_emb = id2embedding[test_id]
            score = cos_sim_func(enroll_emb, test_emb).item()

            if "score_norm" in config:
                # Getting norm stats for enroll impostors
                enroll_rep = paddle.tile(
                    enroll_emb, repeat_times=[train_cohort.shape[0], 1])
                score_e_c = cos_sim_func(enroll_rep, train_cohort)
                if "cohort_size" in config:
                    score_e_c, _ = paddle.topk(
                        score_e_c, k=config.cohort_size, axis=0)
                mean_e_c = paddle.mean(score_e_c, axis=0)
                std_e_c = paddle.std(score_e_c, axis=0)

                # Getting norm stats for test impostors
                test_rep = paddle.tile(
                    test_emb, repeat_times=[train_cohort.shape[0], 1])
                score_t_c = cos_sim_func(test_rep, train_cohort)
                if "cohort_size" in config:
                    score_t_c, _ = paddle.topk(
                        score_t_c, k=config.cohort_size, axis=0)
                mean_t_c = paddle.mean(score_t_c, axis=0)
                std_t_c = paddle.std(score_t_c, axis=0)

                if config.score_norm == "s-norm":
                    score_e = (score - mean_e_c) / std_e_c
                    score_t = (score - mean_t_c) / std_t_c

                    score = 0.5 * (score_e + score_t)
                elif config.score_norm == "z-norm":
                    score = (score - mean_e_c) / std_e_c
                elif config.score_norm == "t-norm":
                    score = (score - mean_t_c) / std_t_c

            scores.append(score)

    return scores, labels


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

    enroll_dataset = CSVDataset(
        os.path.join(args.data_dir, "vox/csv/enroll.csv"),
        feat_type='melspectrogram',
        random_chunk=False,
        n_mels=config.n_mels,
        window_size=config.window_size,
        hop_length=config.hop_size)
    enroll_sampler = BatchSampler(
        enroll_dataset, batch_size=config.batch_size,
        shuffle=False)  # Shuffle to make embedding normalization more robust.
    enroll_loader = DataLoader(enroll_dataset,
                    batch_sampler=enroll_sampler,
                    collate_fn=lambda x: batch_feature_normalize(
                                x, mean_norm=True, std_norm=False),
                    num_workers=config.num_workers,
                    return_list=True,)
    test_dataset = CSVDataset(
        os.path.join(args.data_dir, "vox/csv/test.csv"),
        feat_type='melspectrogram',
        random_chunk=False,
        n_mels=config.n_mels,
        window_size=config.window_size,
        hop_length=config.hop_size)

    test_sampler = BatchSampler(
        test_dataset, batch_size=config.batch_size, shuffle=False)
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

    # stage7: Compute embeddings of audios in enrol and test dataset from model.

    if config.global_embedding_norm:
        mean_var_norm_emb = InputNormalization(
            norm_type="global",
            mean_norm=config.embedding_mean_norm,
            std_norm=config.embedding_std_norm)

    if "score_norm" in config:
        logger.info(f"we will do score norm: {config.score_norm}")
        train_dataset = CSVDataset(
            os.path.join(args.data_dir, "vox/csv/train.csv"),
            feat_type='melspectrogram',
            n_train_snts=config.n_train_snts,
            random_chunk=False,
            n_mels=config.n_mels,
            window_size=config.window_size,
            hop_length=config.hop_size)
        train_sampler = BatchSampler(
            train_dataset, batch_size=config.batch_size, shuffle=False)
        train_loader = DataLoader(train_dataset,
                            batch_sampler=train_sampler,
                            collate_fn=lambda x: batch_feature_normalize(
                                x, mean_norm=True, std_norm=False),
                            num_workers=config.num_workers,
                            return_list=True,)

    id2embedding = {}
    # Run multi times to make embedding normalization more stable.
    logger.info("First loop for enroll and test dataset")
    compute_dataset_embedding(enroll_loader, model, mean_var_norm_emb, config,
                              id2embedding)
    compute_dataset_embedding(test_loader, model, mean_var_norm_emb, config,
                              id2embedding)

    logger.info("Second loop for enroll and test dataset")
    compute_dataset_embedding(enroll_loader, model, mean_var_norm_emb, config,
                              id2embedding)
    compute_dataset_embedding(test_loader, model, mean_var_norm_emb, config,
                              id2embedding)
    mean_var_norm_emb.save(
        os.path.join(args.load_checkpoint, "mean_var_norm_emb"))

    # stage 8: Compute cosine scores.
    train_cohort = None
    if "score_norm" in config:
        train_embeddings = {}
        # cohort embedding not do mean and std norm
        compute_dataset_embedding(train_loader, model, None, config,
                                  train_embeddings)
        train_cohort = paddle.stack(list(train_embeddings.values()))

    # compute the scores
    scores, labels = compute_verification_scores(id2embedding, train_cohort,
                                                 config)

    # compute the EER and threshold
    scores = paddle.to_tensor(scores)
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
