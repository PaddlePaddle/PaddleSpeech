# Copyright (c) 2022 SpeechBrain Authors. All Rights Reserved.
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
"""A popular speaker recognition/diarization model (LDA and PLDA).

Relevant Papers
 - This implementation of PLDA is based on the following papers.

 - PLDA model Training
    * Ye Jiang et. al, "PLDA Modeling in I-Vector and Supervector Space for Speaker Verification," in Interspeech, 2012.
    * Patrick Kenny et. al, "PLDA for speaker verification with utterances of arbitrary duration," in ICASSP, 2013.

 - PLDA scoring (fast scoring)
    * Daniel Garcia-Romero et. al, “Analysis of i-vector length normalization in speaker recognition systems,” in Interspeech, 2011.
    * Weiwei-LIN et. al, "Fast Scoring for PLDA with Uncertainty Propagation," in Odyssey, 2016.
    * Kong Aik Lee et. al, "Multi-session PLDA Scoring of I-vector for Partially Open-Set Speaker Detection," in Interspeech 2013.

Credits
    This code is adapted from: https://git-lium.univ-lemans.fr/Larcher/sidekit
"""
import copy
import pickle

import numpy
from scipy import linalg

from paddlespeech.vector.cluster.diarization import EmbeddingMeta


def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c


class Ndx:
    """
    A class that encodes trial index information.  It has a list of
    model names and a list of test segment names and a matrix
    indicating which combinations of model and test segment are
    trials of interest.

    Arguments
    ---------
    modelset : list
        List of unique models in a ndarray.
    segset : list
        List of unique test segments in a ndarray.
    trialmask : 2D ndarray of bool.
        Rows correspond to the models and columns to the test segments. True, if the trial is of interest.
    """

    def __init__(self,
                 ndx_file_name="",
                 models=numpy.array([]),
                 testsegs=numpy.array([])):
        """
        Initialize a Ndx object by loading information from a file.

        Arguments
        ---------
        ndx_file_name : str
            Name of the file to load.
        """
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.trialmask = numpy.array([], dtype="bool")

        if ndx_file_name == "":
            # This is needed to make sizes same
            d = models.shape[0] - testsegs.shape[0]
            if d != 0:
                if d > 0:
                    last = str(testsegs[-1])
                    pad = numpy.array([last] * d)
                    testsegs = numpy.hstack((testsegs, pad))
                    # pad = testsegs[-d:]
                    # testsegs = numpy.concatenate((testsegs, pad), axis=1)
                else:
                    d = abs(d)
                    last = str(models[-1])
                    pad = numpy.array([last] * d)
                    models = numpy.hstack((models, pad))
                    # pad = models[-d:]
                    # models = numpy.concatenate((models, pad), axis=1)

            modelset = numpy.unique(models)
            segset = numpy.unique(testsegs)

            trialmask = numpy.zeros(
                (modelset.shape[0], segset.shape[0]), dtype="bool")
            for m in range(modelset.shape[0]):
                segs = testsegs[numpy.array(ismember(models, modelset[m]))]
                trialmask[m, ] = ismember(segset, segs)  # noqa E231

            self.modelset = modelset
            self.segset = segset
            self.trialmask = trialmask
            assert self.validate(), "Wrong Ndx format"

        else:
            ndx = Ndx.read(ndx_file_name)
            self.modelset = ndx.modelset
            self.segset = ndx.segset
            self.trialmask = ndx.trialmask

    def save_ndx_object(self, output_file_name):
        with open(output_file_name, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def filter(self, modlist, seglist, keep):
        """
        Removes some of the information in an Ndx. Useful for creating a
        gender specific Ndx from a pooled gender Ndx.  Depending on the
        value of \'keep\', the two input lists indicate the strings to
        retain or the strings to discard.

        Arguments
        ---------
        modlist : array
            A cell array of strings which will be compared with the modelset of 'inndx'.
        seglist : array
            A cell array of strings which will be compared with the segset of 'inndx'.
        keep : bool
            Indicating whether modlist and seglist are the models to keep or discard.
        """
        if keep:
            keepmods = modlist
            keepsegs = seglist
        else:
            keepmods = diff(self.modelset, modlist)
            keepsegs = diff(self.segset, seglist)

        keepmodidx = numpy.array(ismember(self.modelset, keepmods))
        keepsegidx = numpy.array(ismember(self.segset, keepsegs))

        outndx = Ndx()
        outndx.modelset = self.modelset[keepmodidx]
        outndx.segset = self.segset[keepsegidx]
        tmp = self.trialmask[numpy.array(keepmodidx), :]
        outndx.trialmask = tmp[:, numpy.array(keepsegidx)]

        assert outndx.validate, "Wrong Ndx format"

        if self.modelset.shape[0] > outndx.modelset.shape[0]:
            print(
                "Number of models reduced from %d to %d" %
                self.modelset.shape[0],
                outndx.modelset.shape[0], )
        if self.segset.shape[0] > outndx.segset.shape[0]:
            print(
                "Number of test segments reduced from %d to %d",
                self.segset.shape[0],
                outndx.segset.shape[0], )
        return outndx

    def validate(self):
        """
        Checks that an object of type Ndx obeys certain rules that
        must always be true. Returns a boolean value indicating whether the object is valid
        """
        ok = isinstance(self.modelset, numpy.ndarray)
        ok &= isinstance(self.segset, numpy.ndarray)
        ok &= isinstance(self.trialmask, numpy.ndarray)

        ok &= self.modelset.ndim == 1
        ok &= self.segset.ndim == 1
        ok &= self.trialmask.ndim == 2

        ok &= self.trialmask.shape == (self.modelset.shape[0],
                                       self.segset.shape[0], )
        return ok


class Scores:
    """
    A class for storing scores for trials.  The modelset and segset
    fields are lists of model and test segment names respectively.
    The element i,j of scoremat and scoremask corresponds to the
    trial involving model i and test segment j.

    Arguments
    ---------
    modelset : list
        List of unique models in a ndarray.
    segset : list
        List of unique test segments in a ndarray.
    scoremask : 2D ndarray of bool
        Indicates the trials of interest, i.e.,
        the entry i,j in scoremat should be ignored if scoremask[i,j] is False.
    scoremat : 2D ndarray
        Scores matrix.
    """

    def __init__(self, scores_file_name=""):
        """ 
        Initialize a Scores object by loading information from a file HDF5 format.

        Arguments
        ---------
        scores_file_name : str
            Name of the file to load.
        """
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.scoremask = numpy.array([], dtype="bool")
        self.scoremat = numpy.array([])

        if scores_file_name == "":
            pass
        else:
            tmp = Scores.read(scores_file_name)
            self.modelset = tmp.modelset
            self.segset = tmp.segset
            self.scoremask = tmp.scoremask
            self.scoremat = tmp.scoremat

    def __repr__(self):
        ch = "modelset:\n"
        ch += self.modelset + "\n"
        ch += "segset:\n"
        ch += self.segset + "\n"
        ch += "scoremask:\n"
        ch += self.scoremask.__repr__() + "\n"
        ch += "scoremat:\n"
        ch += self.scoremat.__repr__() + "\n"


def fa_model_loop(
        batch_start,
        mini_batch_indices,
        factor_analyser,
        stat0,
        stats,
        e_h,
        e_hh, ):
    """
    A function for PLDA estimation.

    Arguments
    ---------
    batch_start : int
        Index to start at in the list.
    mini_batch_indices : list
        Indices of the elements in the list (should start at zero).
    factor_analyser : instance of PLDA class
        PLDA class object.
    stat0 : tensor
        Matrix of zero-order statistics.
    stats: tensor
        Matrix of first-order statistics.
    e_h : tensor
        An accumulator matrix.
    e_hh: tensor
        An accumulator matrix.
    """
    rank = factor_analyser.F.shape[1]
    if factor_analyser.Sigma.ndim == 2:
        A = factor_analyser.F.T.dot(factor_analyser.F)
        inv_lambda_unique = dict()
        for sess in numpy.unique(stat0[:, 0]):
            inv_lambda_unique[sess] = linalg.inv(sess * A + numpy.eye(A.shape[
                0]))

    tmp = numpy.zeros(
        (factor_analyser.F.shape[1], factor_analyser.F.shape[1]),
        dtype=numpy.float64, )

    for idx in mini_batch_indices:
        if factor_analyser.Sigma.ndim == 1:
            inv_lambda = linalg.inv(
                numpy.eye(rank) + (factor_analyser.F.T * stat0[
                    idx + batch_start, :]).dot(factor_analyser.F))
        else:
            inv_lambda = inv_lambda_unique[stat0[idx + batch_start, 0]]

        aux = factor_analyser.F.T.dot(stats[idx + batch_start, :])
        numpy.dot(aux, inv_lambda, out=e_h[idx])
        e_hh[idx] = inv_lambda + numpy.outer(e_h[idx], e_h[idx], tmp)


def _check_missing_model(enroll, test, ndx):
    # Remove missing models and test segments
    clean_ndx = ndx.filter(enroll.modelset, test.segset, True)

    # Align EmbeddingMeta to match the clean_ndx
    enroll.align_models(clean_ndx.modelset)
    test.align_segments(clean_ndx.segset)

    return clean_ndx


class PLDA:
    """
    A class to train PLDA model from embeddings.

    The input is in paddlespeech.vector.cluster.diarization.EmbeddingMeta format.
    Trains a simplified PLDA model no within-class covariance matrix but full residual covariance matrix.

    Arguments
    ---------
    mean : tensor
        Mean of the vectors.
    F : tensor
        Eigenvoice matrix.
    Sigma : tensor
        Residual matrix.
    """

    def __init__(
            self,
            mean=None,
            F=None,
            Sigma=None,
            rank_f=100,
            nb_iter=10,
            scaling_factor=1.0, ):
        self.mean = None
        self.F = None
        self.Sigma = None
        self.rank_f = rank_f
        self.nb_iter = nb_iter
        self.scaling_factor = scaling_factor

        if mean is not None:
            self.mean = mean
        if F is not None:
            self.F = F
        if Sigma is not None:
            self.Sigma = Sigma

    def plda(
            self,
            emb_meta=None,
            output_file_name=None, ):
        """
        Trains PLDA model with no within class covariance matrix but full residual covariance matrix.

        Arguments
        ---------
        emb_meta : paddlespeech.vector.cluster.diarization.EmbeddingMeta
            Contains vectors and meta-information to perform PLDA
        rank_f : int
            Rank of the between-class covariance matrix.
        nb_iter : int
            Number of iterations to run.
        scaling_factor : float
            Scaling factor to downscale statistics (value between 0 and 1).
        output_file_name : str
            Name of the output file where to store PLDA model.
        """

        # Dimension of the vector (x-vectors stored in stats)
        vect_size = emb_meta.stats.shape[1]

        # Initialize mean and residual covariance from the training data
        self.mean = emb_meta.get_mean_stats()
        self.Sigma = emb_meta.get_total_covariance_stats()

        # Sum stat0 and stat1 for each speaker model
        model_shifted_stat, session_per_model = emb_meta.sum_stat_per_model()

        # Number of speakers (classes) in training set
        class_nb = model_shifted_stat.modelset.shape[0]

        # Multiply statistics by scaling_factor
        model_shifted_stat.stat0 *= self.scaling_factor
        model_shifted_stat.stats *= self.scaling_factor
        session_per_model *= self.scaling_factor

        # Covariance for stats
        sigma_obs = emb_meta.get_total_covariance_stats()
        evals, evecs = linalg.eigh(sigma_obs)

        # Initial F (eigen voice matrix) from rank
        idx = numpy.argsort(evals)[::-1]
        evecs = evecs.real[:, idx[:self.rank_f]]
        self.F = evecs[:, :self.rank_f]

        # Estimate PLDA model by iterating the EM algorithm
        for it in range(self.nb_iter):

            # E-step

            # Copy stats as they will be whitened with a different Sigma for each iteration
            local_stat = copy.deepcopy(model_shifted_stat)

            # Whiten statistics (with the new mean and Sigma)
            local_stat.whiten_stats(self.mean, self.Sigma)

            # Whiten the EigenVoice matrix
            eigen_values, eigen_vectors = linalg.eigh(self.Sigma)
            ind = eigen_values.real.argsort()[::-1]
            eigen_values = eigen_values.real[ind]
            eigen_vectors = eigen_vectors.real[:, ind]
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
            sqr_inv_sigma = numpy.dot(eigen_vectors,
                                      numpy.diag(sqr_inv_eval_sigma))
            self.F = sqr_inv_sigma.T.dot(self.F)

            # Replicate self.stat0
            index_map = numpy.zeros(vect_size, dtype=int)
            _stat0 = local_stat.stat0[:, index_map]

            e_h = numpy.zeros((class_nb, self.rank_f))
            e_hh = numpy.zeros((class_nb, self.rank_f, self.rank_f))

            # loop on model id's
            fa_model_loop(
                batch_start=0,
                mini_batch_indices=numpy.arange(class_nb),
                factor_analyser=self,
                stat0=_stat0,
                stats=local_stat.stats,
                e_h=e_h,
                e_hh=e_hh, )

            # Accumulate for minimum divergence step
            _R = numpy.sum(e_hh, axis=0) / session_per_model.shape[0]

            _C = e_h.T.dot(local_stat.stats).dot(linalg.inv(sqr_inv_sigma))
            _A = numpy.einsum("ijk,i->jk", e_hh, local_stat.stat0.squeeze())

            # M-step
            self.F = linalg.solve(_A, _C).T

            # Update the residual covariance
            self.Sigma = sigma_obs - self.F.dot(_C) / session_per_model.sum()

            # Minimum Divergence step
            self.F = self.F.dot(linalg.cholesky(_R))

    def scoring(
            self,
            enroll,
            test,
            ndx,
            test_uncertainty=None,
            Vtrans=None,
            p_known=0.0,
            scaling_factor=1.0,
            check_missing=True, ):
        """
        Compute the PLDA scores between to sets of vectors. The list of
        trials to perform is given in an Ndx object. PLDA matrices have to be
        pre-computed. i-vectors/x-vectors are supposed to be whitened before.

        Arguments
        ---------
        enroll : paddlespeech.vector.cluster.diarization.EmbeddingMeta
            A EmbeddingMeta in which stats are xvectors.
        test : paddlespeech.vector.cluster.diarization.EmbeddingMeta
            A EmbeddingMeta in which stats are xvectors.
        ndx : paddlespeech.vector.cluster.plda.Ndx
            An Ndx object defining the list of trials to perform.
        p_known : float
            Probability of having a known speaker for open-set
            identification case (=1 for the verification task and =0 for the
            closed-set case).
        check_missing : bool
            If True, check that all models and segments exist.
        """

        enroll_ctr = copy.deepcopy(enroll)
        test_ctr = copy.deepcopy(test)

        # Remove missing models and test segments
        if check_missing:
            clean_ndx = _check_missing_model(enroll_ctr, test_ctr, ndx)
        else:
            clean_ndx = ndx

        # Center the i-vectors around the PLDA mean
        enroll_ctr.center_stats(self.mean)
        test_ctr.center_stats(self.mean)

        # Compute constant component of the PLDA distribution
        invSigma = linalg.inv(self.Sigma)
        I_spk = numpy.eye(self.F.shape[1], dtype="float")

        K = self.F.T.dot(invSigma * scaling_factor).dot(self.F)
        K1 = linalg.inv(K + I_spk)
        K2 = linalg.inv(2 * K + I_spk)

        # Compute the Gaussian distribution constant
        alpha1 = numpy.linalg.slogdet(K1)[1]
        alpha2 = numpy.linalg.slogdet(K2)[1]
        plda_cst = alpha2 / 2.0 - alpha1

        # Compute intermediate matrices
        Sigma_ac = numpy.dot(self.F, self.F.T)
        Sigma_tot = Sigma_ac + self.Sigma
        Sigma_tot_inv = linalg.inv(Sigma_tot)

        Tmp = linalg.inv(Sigma_tot - Sigma_ac.dot(Sigma_tot_inv).dot(Sigma_ac))
        Phi = Sigma_tot_inv - Tmp
        Psi = Sigma_tot_inv.dot(Sigma_ac).dot(Tmp)

        # Compute the different parts of PLDA score
        model_part = 0.5 * numpy.einsum("ij, ji->i",
                                        enroll_ctr.stats.dot(Phi),
                                        enroll_ctr.stats.T)
        seg_part = 0.5 * numpy.einsum("ij, ji->i",
                                      test_ctr.stats.dot(Phi), test_ctr.stats.T)

        # Compute verification scores
        score = Scores()  # noqa F821
        score.modelset = clean_ndx.modelset
        score.segset = clean_ndx.segset
        score.scoremask = clean_ndx.trialmask

        score.scoremat = model_part[:, numpy.newaxis] + seg_part + plda_cst
        score.scoremat += enroll_ctr.stats.dot(Psi).dot(test_ctr.stats.T)
        score.scoremat *= scaling_factor

        # Case of open-set identification, we compute the log-likelihood
        # by taking into account the probability of having a known impostor
        # or an out-of set class
        if p_known != 0:
            N = score.scoremat.shape[0]
            open_set_scores = numpy.empty(score.scoremat.shape)
            tmp = numpy.exp(score.scoremat)
            for ii in range(N):
                # open-set term
                open_set_scores[ii, :] = score.scoremat[ii, :] - numpy.log(
                    p_known * tmp[~(numpy.arange(N) == ii)].sum(axis=0) / (
                        N - 1) + (1 - p_known))
            score.scoremat = open_set_scores

        return score


if __name__ == '__main__':
    import random

    dim, N, n_spkrs = 10, 100, 10
    train_xv = numpy.random.rand(N, dim)
    md = ['md' + str(random.randrange(1, n_spkrs, 1)) for i in range(N)]  # spk
    modelset = numpy.array(md, dtype="|O")
    sg = ['sg' + str(i) for i in range(N)]  # utt
    segset = numpy.array(sg, dtype="|O")
    stat0 = numpy.array([[1.0]] * N)
    xvectors_stat = EmbeddingMeta(
        modelset=modelset, segset=segset, stats=train_xv)
    # Training PLDA model: M ~ (mean, F, Sigma)
    plda = PLDA(rank_f=5)
    plda.plda(xvectors_stat)
    print(plda.mean.shape)  #(10,)
    print(plda.F.shape)  #(10, 5)
    print(plda.Sigma.shape)  #(10, 10)
    # Enrollment (20 utts),
    en_N = 20
    en_xv = numpy.random.rand(en_N, dim)
    en_sgs = ['en' + str(i) for i in range(en_N)]
    en_sets = numpy.array(en_sgs, dtype="|O")
    en_stat = EmbeddingMeta(modelset=en_sets, segset=en_sets, stats=en_xv)
    # Test (30 utts)
    te_N = 30
    te_xv = numpy.random.rand(te_N, dim)
    te_sgs = ['te' + str(i) for i in range(te_N)]
    te_sets = numpy.array(te_sgs, dtype="|O")
    te_stat = EmbeddingMeta(modelset=te_sets, segset=te_sets, stats=te_xv)
    ndx = Ndx(models=en_sets, testsegs=te_sets)
    # PLDA Scoring
    scores_plda = plda.scoring(en_stat, te_stat, ndx)
    print(scores_plda.scoremat.shape)  #(20, 30)
