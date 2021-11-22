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
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


class LSTMSpeakerEncoder(nn.Layer):
    def __init__(self, n_mels, num_layers, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(n_mels, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.similarity_weight = self.create_parameter(
            [1], default_initializer=I.Constant(10.))
        self.similarity_bias = self.create_parameter(
            [1], default_initializer=I.Constant(-5.))

    def forward(self, utterances, num_speakers, initial_states=None):
        normalized_embeds = self.embed_sequences(utterances, initial_states)
        embeds = normalized_embeds.reshape([num_speakers, -1, num_speakers])
        loss, eer = self.loss(embeds)
        return loss, eer

    def embed_sequences(self, utterances, initial_states=None, reduce=False):
        out, (h, c) = self.lstm(utterances, initial_states)
        embeds = F.relu(self.linear(h[-1]))
        normalized_embeds = F.normalize(embeds)
        if reduce:
            embed = paddle.mean(normalized_embeds, 0)
            embed = F.normalize(embed, axis=0)
            return embed
        return normalized_embeds

    def embed_utterance(self, utterances, initial_states=None):
        # utterances: [B, T, C] -> embed [C']
        embed = self.embed_sequences(utterances, initial_states, reduce=True)
        return embed

    def similarity_matrix(self, embeds):
        # (N, M, C)
        speakers_per_batch, utterances_per_speaker, embed_dim = embeds.shape

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = paddle.mean(embeds, axis=1)
        centroids_incl_norm = paddle.norm(
            centroids_incl, p=2, axis=1, keepdim=True)
        normalized_centroids_incl = centroids_incl / centroids_incl_norm

        # Exclusive centroids (1 per utterance)
        centroids_excl = paddle.broadcast_to(
            paddle.sum(embeds, axis=1, keepdim=True), embeds.shape) - embeds
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl_norm = paddle.norm(
            centroids_excl, p=2, axis=2, keepdim=True)
        normalized_centroids_excl = centroids_excl / centroids_excl_norm

        p1 = paddle.matmul(
            embeds.reshape([-1, embed_dim]),
            normalized_centroids_incl,
            transpose_y=True)  # (NMN)
        p1 = p1.reshape([-1])
        # print("p1: ", p1.shape)
        p2 = paddle.bmm(
            embeds.reshape([-1, 1, embed_dim]),
            normalized_centroids_excl.reshape([-1, embed_dim, 1]))  # (NM, 1, 1)
        p2 = p2.reshape([-1])  # （NM)

        # begin: alternative implementation for scatter
        with paddle.no_grad():
            index = paddle.arange(
                0, speakers_per_batch * utterances_per_speaker,
                dtype="int64").reshape(
                    [speakers_per_batch, utterances_per_speaker])
            index = index * speakers_per_batch + paddle.arange(
                0, speakers_per_batch, dtype="int64").unsqueeze(-1)
            index = paddle.reshape(index, [-1])
        ones = paddle.ones(
            [speakers_per_batch * utterances_per_speaker * speakers_per_batch])
        zeros = paddle.zeros_like(index, dtype=ones.dtype)
        mask_p1 = paddle.scatter(ones, index, zeros)
        p = p1 * mask_p1 + (1 - mask_p1) * paddle.scatter(ones, index, p2)
        # end: alternative implementation for scatter
        # p = paddle.scatter(p1, index, p2)

        p = p * self.similarity_weight + self.similarity_bias  # neg
        p = p.reshape(
            [speakers_per_batch * utterances_per_speaker, speakers_per_batch])
        return p, p1, p2

    def do_gradient_ops(self):
        for p in [self.similarity_weight, self.similarity_bias]:
            g = p._grad_ivar()
            g = g * 0.01

    def inv_argmax(self, i, num):
        return np.eye(1, num, i, dtype=int)[0]

    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Loss
        sim_matrix, *_ = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape(
            [speakers_per_batch * utterances_per_speaker, speakers_per_batch])
        target = paddle.arange(
            0, speakers_per_batch, dtype="int64").unsqueeze(-1)
        target = paddle.expand(target,
                               [speakers_per_batch, utterances_per_speaker])
        target = paddle.reshape(target, [-1])

        loss = nn.CrossEntropyLoss()(sim_matrix, target)

        # EER (not backpropagated)
        with paddle.no_grad():
            ground_truth = target.numpy()
            labels = np.array(
                [self.inv_argmax(i, speakers_per_batch) for i in ground_truth])
            preds = sim_matrix.numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return loss, eer
