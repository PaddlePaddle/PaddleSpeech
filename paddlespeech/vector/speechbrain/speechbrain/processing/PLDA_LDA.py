"""A popular speaker recognition/diarization model (LDA and PLDA).

Authors
 * Anthony Larcher 2020
 * Nauman Dawalatabad 2020

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

import numpy
import copy
import pickle

from scipy import linalg

STAT_TYPE = numpy.float64


class StatObject_SB:
    """A utility class for PLDA class used for statistics calculations.

    This is also used to pack deep embeddings and meta-information in one object.

    Arguments
    ---------
    modelset : list
        List of model IDs for each session as an array of strings.
    segset : list
        List of session IDs as an array of strings.
    start : int
        Index of the first frame of the segment.
    stop : int
        Index of the last frame of the segment.
    stat0 : tensor
        An ndarray of float64. Each line contains 0-th order statistics
        from the corresponding session.
    stat1 : tensor
        An ndarray of float64. Each line contains 1-st order statistics
        from the corresponding session.
    """

    def __init__(
        self,
        modelset=None,
        segset=None,
        start=None,
        stop=None,
        stat0=None,
        stat1=None,
    ):

        if modelset is None:  # For creating empty stat server
            self.modelset = numpy.empty(0, dtype="|O")
            self.segset = numpy.empty(0, dtype="|O")
            self.start = numpy.empty(0, dtype="|O")
            self.stop = numpy.empty(0, dtype="|O")
            self.stat0 = numpy.array([], dtype=STAT_TYPE)
            self.stat1 = numpy.array([], dtype=STAT_TYPE)
        else:
            self.modelset = modelset
            self.segset = segset
            self.start = start
            self.stop = stop
            self.stat0 = stat0
            self.stat1 = stat1

    def __repr__(self):
        ch = "-" * 30 + "\n"
        ch += "modelset: " + self.modelset.__repr__() + "\n"
        ch += "segset: " + self.segset.__repr__() + "\n"
        ch += "seg start:" + self.start.__repr__() + "\n"
        ch += "seg stop:" + self.stop.__repr__() + "\n"
        ch += "stat0:" + self.stat0.__repr__() + "\n"
        ch += "stat1:" + self.stat1.__repr__() + "\n"
        ch += "-" * 30 + "\n"
        return ch

    def save_stat_object(self, filename):
        with open(filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def get_model_segsets(self, mod_id):
        """Return segments of a given model.

        Arguments
        ---------
        mod_id : str
            ID of the model for which segments will be returned.
        """
        return self.segset[self.modelset == mod_id]

    def get_model_start(self, mod_id):
        """Return start of segment of a given model.

        Arguments
        ---------
        mod_id : str
            ID of the model for which start will be returned.
        """
        return self.start[self.modelset == mod_id]

    def get_model_stop(self, mod_id):
        """Return stop of segment of a given model.

        Arguments
        ---------
        mod_id : str
            ID of the model which stop will be returned.
        """
        return self.stop[self.modelset == mod_id]

    def get_mean_stat1(self):
        """Return the mean of first order statistics.
        """
        mu = numpy.mean(self.stat1, axis=0)
        return mu

    def get_total_covariance_stat1(self):
        """Compute and return the total covariance matrix of the first-order
            statistics.
        """
        C = self.stat1 - self.stat1.mean(axis=0)
        return numpy.dot(C.transpose(), C) / self.stat1.shape[0]

    def get_model_stat0(self, mod_id):
        """Return zero-order statistics of a given model

        Arguments
        ---------
        mod_id : str
            ID of the model which stat0 will be returned.
        """
        S = self.stat0[self.modelset == mod_id, :]
        return S

    def get_model_stat1(self, mod_id):
        """Return first-order statistics of a given model.

        Arguments
        ---------
        mod_id : str
            ID of the model which stat1 will be returned.
        """
        return self.stat1[self.modelset == mod_id, :]

    def sum_stat_per_model(self):
        """Sum the zero- and first-order statistics per model and store them
        in a new StatObject_SB.
        Returns a StatObject_SB object with the statistics summed per model
        and a numpy array with session_per_model.
        """

        sts_per_model = StatObject_SB()
        sts_per_model.modelset = numpy.unique(
            self.modelset
        )  # nd: get uniq spkr ids
        sts_per_model.segset = copy.deepcopy(sts_per_model.modelset)
        sts_per_model.stat0 = numpy.zeros(
            (sts_per_model.modelset.shape[0], self.stat0.shape[1]),
            dtype=STAT_TYPE,
        )
        sts_per_model.stat1 = numpy.zeros(
            (sts_per_model.modelset.shape[0], self.stat1.shape[1]),
            dtype=STAT_TYPE,
        )

        # Keep this. may need this in future (Nauman)
        # sts_per_model.start = numpy.empty(
        #    sts_per_model.segset.shape, "|O"
        # )  # ndf: restructure this
        # sts_per_model.stop = numpy.empty(sts_per_model.segset.shape, "|O")

        session_per_model = numpy.zeros(numpy.unique(self.modelset).shape[0])

        # For each model sum the stats
        for idx, model in enumerate(sts_per_model.modelset):
            sts_per_model.stat0[idx, :] = self.get_model_stat0(model).sum(
                axis=0
            )
            sts_per_model.stat1[idx, :] = self.get_model_stat1(model).sum(
                axis=0
            )
            session_per_model[idx] += self.get_model_stat1(model).shape[0]
        return sts_per_model, session_per_model

    def center_stat1(self, mu):
        """Center first order statistics.

        Arguments
        ---------
        mu : array
            Array to center on.
        """

        dim = self.stat1.shape[1] / self.stat0.shape[1]
        index_map = numpy.repeat(numpy.arange(self.stat0.shape[1]), dim)
        self.stat1 = self.stat1 - (
            self.stat0[:, index_map] * mu.astype(STAT_TYPE)
        )

    def norm_stat1(self):
        """Divide all first-order statistics by their Euclidean norm.
        """

        vect_norm = numpy.clip(
            numpy.linalg.norm(self.stat1, axis=1), 1e-08, numpy.inf
        )
        self.stat1 = (self.stat1.transpose() / vect_norm).transpose()

    def rotate_stat1(self, R):
        """Rotate first-order statistics by a right-product.

        Arguments
        ---------
        R : ndarray
            Matrix to use for right product on the first order statistics.
        """
        self.stat1 = numpy.dot(self.stat1, R)

    def whiten_stat1(self, mu, sigma, isSqrInvSigma=False):
        """Whiten first-order statistics
        If sigma.ndim == 1, case of a diagonal covariance.
        If sigma.ndim == 2, case of a single Gaussian with full covariance.
        If sigma.ndim == 3, case of a full covariance UBM.

        Arguments
        ---------
        mu : array
            Mean vector to be subtracted from the statistics.
        sigma : narray
            Co-variance matrix or covariance super-vector.
        isSqrInvSigma : bool
            True if the input Sigma matrix is the inverse of the square root of a covariance matrix.
        """

        if sigma.ndim == 1:
            self.center_stat1(mu)
            self.stat1 = self.stat1 / numpy.sqrt(sigma.astype(STAT_TYPE))

        elif sigma.ndim == 2:
            # Compute the inverse square root of the co-variance matrix Sigma
            sqr_inv_sigma = sigma

            if not isSqrInvSigma:
                # eigen_values, eigen_vectors = scipy.linalg.eigh(sigma)
                eigen_values, eigen_vectors = linalg.eigh(sigma)
                ind = eigen_values.real.argsort()[::-1]
                eigen_values = eigen_values.real[ind]
                eigen_vectors = eigen_vectors.real[:, ind]

                sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
                sqr_inv_sigma = numpy.dot(
                    eigen_vectors, numpy.diag(sqr_inv_eval_sigma)
                )
            else:
                pass

            # Whitening of the first-order statistics
            self.center_stat1(mu)  # CENTERING
            self.rotate_stat1(sqr_inv_sigma)

        elif sigma.ndim == 3:
            # we assume that sigma is a 3D ndarray of size D x n x n
            # where D is the number of distributions and n is the dimension of a single distribution
            n = self.stat1.shape[1] // self.stat0.shape[1]
            sess_nb = self.stat0.shape[0]
            self.center_stat1(mu)
            self.stat1 = (
                numpy.einsum(
                    "ikj,ikl->ilj", self.stat1.T.reshape(-1, n, sess_nb), sigma
                )
                .reshape(-1, sess_nb)
                .T
            )

        else:
            raise Exception("Wrong dimension of Sigma, must be 1 or 2")

    def align_models(self, model_list):
        """Align models of the current StatServer to match a list of models
            provided as input parameter. The size of the StatServer might be
            reduced to match the input list of models.

        Arguments
        ---------
        model_list : ndarray of strings
            List of models to match.
        """
        indx = numpy.array(
            [numpy.argwhere(self.modelset == v)[0][0] for v in model_list]
        )
        self.segset = self.segset[indx]
        self.modelset = self.modelset[indx]
        self.start = self.start[indx]
        self.stop = self.stop[indx]
        self.stat0 = self.stat0[indx, :]
        self.stat1 = self.stat1[indx, :]

    def align_segments(self, segment_list):
        """Align segments of the current StatServer to match a list of segment
            provided as input parameter. The size of the StatServer might be
            reduced to match the input list of segments.

        Arguments
        ---------
        segment_list: ndarray of strings
            list of segments to match
        """
        indx = numpy.array(
            [numpy.argwhere(self.segset == v)[0][0] for v in segment_list]
        )
        self.segset = self.segset[indx]
        self.modelset = self.modelset[indx]
        self.start = self.start[indx]
        self.stop = self.stop[indx]
        self.stat0 = self.stat0[indx, :]
        self.stat1 = self.stat1[indx, :]

    def get_lda_matrix_stat1(self, rank):
        """Compute and return the Linear Discriminant Analysis matrix
            on the first-order statistics. Columns of the LDA matrix are ordered
            according to the corresponding eigenvalues in descending order.

        Arguments
        ---------
        rank : int
            Rank of the LDA matrix to return.
        """

        vect_size = self.stat1.shape[1]
        unique_speaker = numpy.unique(self.modelset)

        mu = self.get_mean_stat1()

        class_means = numpy.zeros((unique_speaker.shape[0], vect_size))
        Sw = numpy.zeros((vect_size, vect_size))

        spk_idx = 0
        for speaker_id in unique_speaker:
            spk_sessions = self.get_model_stat1(speaker_id) - numpy.mean(
                self.get_model_stat1(speaker_id), axis=0
            )
            Sw += (
                numpy.dot(spk_sessions.transpose(), spk_sessions)
                / spk_sessions.shape[0]
            )
            class_means[spk_idx, :] = numpy.mean(
                self.get_model_stat1(speaker_id), axis=0
            )
            spk_idx += 1

        # Compute Between-class scatter matrix
        class_means = class_means - mu
        Sb = numpy.dot(class_means.transpose(), class_means)

        # Compute the Eigenvectors & eigenvalues of the discrimination matrix
        DiscriminationMatrix = numpy.dot(Sb, linalg.inv(Sw)).transpose()
        eigen_values, eigen_vectors = linalg.eigh(DiscriminationMatrix)
        eigen_values = eigen_values.real
        eigen_vectors = eigen_vectors.real

        # Rearrange the eigenvectors according to decreasing eigenvalues
        # get indexes of the rank top eigen values
        idx = eigen_values.real.argsort()[-rank:][::-1]
        L = eigen_vectors[:, idx]
        return L


def diff(list1, list2):
    c = [item for item in list1 if item not in list2]
    c.sort()
    return c


def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c


class Ndx:
    """A class that encodes trial index information.  It has a list of
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

    def __init__(
        self, ndx_file_name="", models=numpy.array([]), testsegs=numpy.array([])
    ):
        """Initialize a Ndx object by loading information from a file.

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
                (modelset.shape[0], segset.shape[0]), dtype="bool"
            )
            for m in range(modelset.shape[0]):
                segs = testsegs[numpy.array(ismember(models, modelset[m]))]
                trialmask[m,] = ismember(segset, segs)  # noqa E231

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
        """Removes some of the information in an Ndx. Useful for creating a
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
                "Number of models reduced from %d to %d"
                % self.modelset.shape[0],
                outndx.modelset.shape[0],
            )
        if self.segset.shape[0] > outndx.segset.shape[0]:
            print(
                "Number of test segments reduced from %d to %d",
                self.segset.shape[0],
                outndx.segset.shape[0],
            )
        return outndx

    def validate(self):
        """Checks that an object of type Ndx obeys certain rules that
        must always be true. Returns a boolean value indicating whether the object is valid
        """
        ok = isinstance(self.modelset, numpy.ndarray)
        ok &= isinstance(self.segset, numpy.ndarray)
        ok &= isinstance(self.trialmask, numpy.ndarray)

        ok &= self.modelset.ndim == 1
        ok &= self.segset.ndim == 1
        ok &= self.trialmask.ndim == 2

        ok &= self.trialmask.shape == (
            self.modelset.shape[0],
            self.segset.shape[0],
        )
        return ok


class Scores:
    """A class for storing scores for trials.  The modelset and segset
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
        """ Initialize a Scores object by loading information from a file HDF5 format.

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


## PLDA and LDA functionalities starts here


def fa_model_loop(
    batch_start, mini_batch_indices, factor_analyser, stat0, stat1, e_h, e_hh,
):
    """A function for PLDA estimation.

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
    stat1: tensor
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
            inv_lambda_unique[sess] = linalg.inv(
                sess * A + numpy.eye(A.shape[0])
            )

    tmp = numpy.zeros(
        (factor_analyser.F.shape[1], factor_analyser.F.shape[1]),
        dtype=numpy.float64,
    )

    for idx in mini_batch_indices:
        if factor_analyser.Sigma.ndim == 1:
            inv_lambda = linalg.inv(
                numpy.eye(rank)
                + (factor_analyser.F.T * stat0[idx + batch_start, :]).dot(
                    factor_analyser.F
                )
            )
        else:
            inv_lambda = inv_lambda_unique[stat0[idx + batch_start, 0]]

        aux = factor_analyser.F.T.dot(stat1[idx + batch_start, :])
        numpy.dot(aux, inv_lambda, out=e_h[idx])
        e_hh[idx] = inv_lambda + numpy.outer(e_h[idx], e_h[idx], tmp)


def _check_missing_model(enroll, test, ndx):
    # Remove missing models and test segments
    clean_ndx = ndx.filter(enroll.modelset, test.segset, True)

    # Align StatServers to match the clean_ndx
    enroll.align_models(clean_ndx.modelset)
    test.align_segments(clean_ndx.segset)

    return clean_ndx


def fast_PLDA_scoring(
    enroll,
    test,
    ndx,
    mu,
    F,
    Sigma,
    test_uncertainty=None,
    Vtrans=None,
    p_known=0.0,
    scaling_factor=1.0,
    check_missing=True,
):
    """Compute the PLDA scores between to sets of vectors. The list of
    trials to perform is given in an Ndx object. PLDA matrices have to be
    pre-computed. i-vectors/x-vectors are supposed to be whitened before.

    Arguments
    ---------
    enroll : speechbrain.utils.Xvector_PLDA_sp.StatObject_SB
        A StatServer in which stat1 are xvectors.
    test : speechbrain.utils.Xvector_PLDA_sp.StatObject_SB
        A StatServer in which stat1 are xvectors.
    ndx : speechbrain.utils.Xvector_PLDA_sp.Ndx
        An Ndx object defining the list of trials to perform.
    mu : double
        The mean vector of the PLDA gaussian.
    F : tensor
        The between-class co-variance matrix of the PLDA.
    Sigma: tensor
        The residual covariance matrix.
    p_known : float
        Probability of having a known speaker for open-set
        identification case (=1 for the verification task and =0 for the
        closed-set case).
    check_missing : bool
        If True, check that all models and segments exist.
    """

    enroll_ctr = copy.deepcopy(enroll)
    test_ctr = copy.deepcopy(test)

    # If models are not unique, compute the mean per model, display a warning
    if not numpy.unique(enroll_ctr.modelset).shape == enroll_ctr.modelset.shape:
        # logging.warning("Enrollment models are not unique, average i-vectors")
        enroll_ctr = enroll_ctr.mean_stat_per_model()

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll_ctr, test_ctr, ndx)
    else:
        clean_ndx = ndx

    # Center the i-vectors around the PLDA mean
    enroll_ctr.center_stat1(mu)
    test_ctr.center_stat1(mu)

    # If models are not unique, compute the mean per model, display a warning
    if not numpy.unique(enroll_ctr.modelset).shape == enroll_ctr.modelset.shape:
        # logging.warning("Enrollment models are not unique, average i-vectors")
        enroll_ctr = enroll_ctr.mean_stat_per_model()

    # Compute constant component of the PLDA distribution
    invSigma = linalg.inv(Sigma)
    I_spk = numpy.eye(F.shape[1], dtype="float")

    K = F.T.dot(invSigma * scaling_factor).dot(F)
    K1 = linalg.inv(K + I_spk)
    K2 = linalg.inv(2 * K + I_spk)

    # Compute the Gaussian distribution constant
    alpha1 = numpy.linalg.slogdet(K1)[1]
    alpha2 = numpy.linalg.slogdet(K2)[1]
    plda_cst = alpha2 / 2.0 - alpha1

    # Compute intermediate matrices
    Sigma_ac = numpy.dot(F, F.T)
    Sigma_tot = Sigma_ac + Sigma
    Sigma_tot_inv = linalg.inv(Sigma_tot)

    Tmp = linalg.inv(Sigma_tot - Sigma_ac.dot(Sigma_tot_inv).dot(Sigma_ac))
    Phi = Sigma_tot_inv - Tmp
    Psi = Sigma_tot_inv.dot(Sigma_ac).dot(Tmp)

    # Compute the different parts of PLDA score
    model_part = 0.5 * numpy.einsum(
        "ij, ji->i", enroll_ctr.stat1.dot(Phi), enroll_ctr.stat1.T
    )
    seg_part = 0.5 * numpy.einsum(
        "ij, ji->i", test_ctr.stat1.dot(Phi), test_ctr.stat1.T
    )

    # Compute verification scores
    score = Scores()  # noqa F821
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask

    score.scoremat = model_part[:, numpy.newaxis] + seg_part + plda_cst
    score.scoremat += enroll_ctr.stat1.dot(Psi).dot(test_ctr.stat1.T)
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
                p_known * tmp[~(numpy.arange(N) == ii)].sum(axis=0) / (N - 1)
                + (1 - p_known)
            )
        score.scoremat = open_set_scores

    return score


class LDA:
    """A class to perform Linear Discriminant Analysis.

    It returns the low dimensional representation as per LDA.

    Arguments
    ---------
    reduced_dim : int
        The dimension of the output representation.
    """

    def __init__(self,):
        self.transform_mat = None

    def do_lda(self, stat_server=None, reduced_dim=2, transform_mat=None):
        """Performs LDA and projects the vectors onto lower dimension space.

        Arguments
        ---------
        stat_server : object of speechbrain.processing.PLDA_LDA.StatObject_SB.
            Contains vectors and meta-information to perform LDA.
        reduced_dim : int
            Dimension of the reduced space.
        """

        # Get transformation matrix and project
        if transform_mat is None:
            self.transform_mat = stat_server.get_lda_matrix_stat1(reduced_dim)
        else:
            self.transform_mat = transform_mat

        # Projection
        new_train_obj = copy.deepcopy(stat_server)
        new_train_obj.rotate_stat1(self.transform_mat)

        return new_train_obj


class PLDA:
    """A class to train PLDA model from embeddings.

    The input is in speechbrain.utils.StatObject_SB format.
    Trains a simplified PLDA model no within-class covariance matrix but full residual covariance matrix.

    Arguments
    ---------
    mean : tensor
        Mean of the vectors.
    F : tensor
        Eigenvoice matrix.
    Sigma : tensor
        Residual matrix.

    Example
    -------
    >>> from speechbrain.processing.PLDA_LDA import *
    >>> import random, numpy
    >>> dim, N = 10, 100
    >>> n_spkrs = 10
    >>> train_xv = numpy.random.rand(N, dim)
    >>> md = ['md'+str(random.randrange(1,n_spkrs,1)) for i in range(N)]
    >>> modelset = numpy.array(md, dtype="|O")
    >>> sg = ['sg'+str(i) for i in range(N)]
    >>> segset = numpy.array(sg, dtype="|O")
    >>> s = numpy.array([None] * N)
    >>> stat0 = numpy.array([[1.0]]* N)
    >>> xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=train_xv)
    >>> # Training PLDA model: M ~ (mean, F, Sigma)
    >>> plda = PLDA(rank_f=5)
    >>> plda.plda(xvectors_stat)
    >>> print (plda.mean.shape)
    (10,)
    >>> print (plda.F.shape)
    (10, 5)
    >>> print (plda.Sigma.shape)
    (10, 10)
    >>> # Enrollment (20 utts), Test (30 utts)
    >>> en_N = 20
    >>> en_xv = numpy.random.rand(en_N, dim)
    >>> en_sgs = ['en'+str(i) for i in range(en_N)]
    >>> en_sets = numpy.array(en_sgs, dtype="|O")
    >>> en_s = numpy.array([None] * en_N)
    >>> en_stat0 = numpy.array([[1.0]]* en_N)
    >>> en_stat = StatObject_SB(modelset=en_sets, segset=en_sets, start=en_s, stop=en_s, stat0=en_stat0, stat1=en_xv)
    >>> te_N = 30
    >>> te_xv = numpy.random.rand(te_N, dim)
    >>> te_sgs = ['te'+str(i) for i in range(te_N)]
    >>> te_sets = numpy.array(te_sgs, dtype="|O")
    >>> te_s = numpy.array([None] * te_N)
    >>> te_stat0 = numpy.array([[1.0]]* te_N)
    >>> te_stat = StatObject_SB(modelset=te_sets, segset=te_sets, start=te_s, stop=te_s, stat0=te_stat0, stat1=te_xv)
    >>> ndx = Ndx(models=en_sets, testsegs=te_sets)
    >>> # PLDA Scoring
    >>> scores_plda = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma)
    >>> print (scores_plda.scoremat.shape)
    (20, 30)
    """

    def __init__(
        self,
        mean=None,
        F=None,
        Sigma=None,
        rank_f=100,
        nb_iter=10,
        scaling_factor=1.0,
    ):
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
        stat_server=None,
        output_file_name=None,
        whiten=False,
        w_stat_server=None,
    ):
        """Trains PLDA model with no within class covariance matrix but full residual covariance matrix.

        Arguments
        ---------
        stat_server : speechbrain.processing.PLDA_LDA.StatObject_SB
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

        # Dimension of the vector (x-vectors stored in stat1)
        vect_size = stat_server.stat1.shape[1]  # noqa F841

        # Whitening (Optional)
        if whiten is True:
            w_mean = w_stat_server.get_mean_stat1()
            w_Sigma = w_stat_server.get_total_covariance_stat1()
            stat_server.whiten_stat1(w_mean, w_Sigma)

        # Initialize mean and residual covariance from the training data
        self.mean = stat_server.get_mean_stat1()
        self.Sigma = stat_server.get_total_covariance_stat1()

        # Sum stat0 and stat1 for each speaker model
        model_shifted_stat, session_per_model = stat_server.sum_stat_per_model()

        # Number of speakers (classes) in training set
        class_nb = model_shifted_stat.modelset.shape[0]

        # Multiply statistics by scaling_factor
        model_shifted_stat.stat0 *= self.scaling_factor
        model_shifted_stat.stat1 *= self.scaling_factor
        session_per_model *= self.scaling_factor

        # Covariance for stat1
        sigma_obs = stat_server.get_total_covariance_stat1()
        evals, evecs = linalg.eigh(sigma_obs)

        # Initial F (eigen voice matrix) from rank
        idx = numpy.argsort(evals)[::-1]
        evecs = evecs.real[:, idx[: self.rank_f]]
        self.F = evecs[:, : self.rank_f]

        # Estimate PLDA model by iterating the EM algorithm
        for it in range(self.nb_iter):

            # E-step
            # print(
            #    f"E-step: Estimate between class covariance, it {it+1} / {nb_iter}"
            # )

            # Copy stats as they will be whitened with a different Sigma for each iteration
            local_stat = copy.deepcopy(model_shifted_stat)

            # Whiten statistics (with the new mean and Sigma)
            local_stat.whiten_stat1(self.mean, self.Sigma)

            # Whiten the EigenVoice matrix
            eigen_values, eigen_vectors = linalg.eigh(self.Sigma)
            ind = eigen_values.real.argsort()[::-1]
            eigen_values = eigen_values.real[ind]
            eigen_vectors = eigen_vectors.real[:, ind]
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
            sqr_inv_sigma = numpy.dot(
                eigen_vectors, numpy.diag(sqr_inv_eval_sigma)
            )
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
                stat1=local_stat.stat1,
                e_h=e_h,
                e_hh=e_hh,
            )

            # Accumulate for minimum divergence step
            _R = numpy.sum(e_hh, axis=0) / session_per_model.shape[0]

            _C = e_h.T.dot(local_stat.stat1).dot(linalg.inv(sqr_inv_sigma))
            _A = numpy.einsum("ijk,i->jk", e_hh, local_stat.stat0.squeeze())

            # M-step
            # print("M-step")
            self.F = linalg.solve(_A, _C).T

            # Update the residual covariance
            self.Sigma = sigma_obs - self.F.dot(_C) / session_per_model.sum()

            # Minimum Divergence step
            self.F = self.F.dot(linalg.cholesky(_R))
