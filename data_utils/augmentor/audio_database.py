from __future__ import print_function
from collections import defaultdict
import bisect
import logging
import numpy as np
import os
import random
import sys

UNK_TAG = "<UNK>"


def stream_audio_index(fname, UNK=UNK_TAG):
    """Reads an audio index file and emits one record in the index at a time.

    :param fname: audio index path
    :type fname: basestring
    :param UNK: UNK token to denote that certain audios are not tagged.
    :type UNK: basesring

    Yields:
        idx, duration, size, relpath, tags (int, float, int, str, list(str)):
            audio file id, length of the audio in seconds, size in byte,
            relative path w.r.t. to the root noise directory, list of tags
    """
    with open(fname) as audio_index_file:
        for i, line in enumerate(audio_index_file):
            tok = line.strip().split("\t")
            assert len(tok) >= 4, \
                "Invalid line at line {} in file {}".format(
                    i + 1, audio_index_file)
            idx = int(tok[0])
            duration = float(tok[1])
            # Sometimes, the duration can round down to 0.0
            assert duration >= 0.0, \
                "Invalid duration at line {} in file {}".format(
                    i + 1, audio_index_file)
            size = int(tok[2])
            assert size > 0, \
                "Invalid size at line {} in file {}".format(
                    i + 1, audio_index_file)
            relpath = tok[3]
            if len(tok) == 4:
                tags = [UNK_TAG]
            else:
                tags = tok[4:]
            yield idx, duration, size, relpath, tags


def truncate_float(val, ndigits=6):
    """ Truncates a floating-point value to have the desired number of
    digits after the decimal point.

    :param val: input value.
    :type val: float
    :parma ndigits: desired number of digits.
    :type ndigits: int

    :return: truncated value
    :rtype: float
    """
    p = 10.0**ndigits
    return float(int(val * p)) / p


def print_audio_index(idx, duration, size, relpath, tags, file=sys.stdout):
    """Prints an audio record to the index file.

    :param idx: Audio file id.
    :type idx: int
    :param duration: length of the audio in seconds
    :type duration: float
    :param size: size of the file in bytes
    :type size: int
    :param relpath: relative path w.r.t. to the root noise directory.
    :type relpath:  basestring
    :parma tags: list of tags
    :parma tags: list(str)
    :parma file: file to which we want to write an audio record.
    :type file: sys.stdout
    """
    file.write("{}\t{:.6f}\t{}\t{}"
               .format(idx, truncate_float(duration, ndigits=6), size, relpath))
    for tag in tags:
        file.write("\t{}".format(tag))
    file.write("\n")


class AudioIndex(object):
    """ In-memory index of audio files that do not have annotations.
    This supports duration-based sampling and sampling from a target
    distribution.

    Each line in the index file consists of the following fields:
        (id (int), duration (float), size (int), relative path (str),
         list of tags ([str]))
    """

    def __init__(self):
        self.audio_dir = None
        self.index_fname = None
        self.tags = None
        self.bin_size = 2.0
        self.clear()

    def clear(self):
        """ Clears the index

        Returns:
            None
        """
        self.idx_to_record = {}
        # The list of indices correspond to audio files whose duration is
        # greater than or equal to the key.
        self.duration_to_id_set = {}
        self.duration_to_id_set_per_tag = defaultdict(lambda: {})
        self.duration_to_list = defaultdict(lambda: [])
        self.duration_to_list_per_tag = defaultdict(
            lambda: defaultdict(lambda: []))
        self.tag_to_id_set = defaultdict(lambda: set())
        self.shared_duration_bins = []
        self.id_set_complete = set()
        self.id_set = set()
        self.duration_bins = []

    def has_audio(self, distr=None):
        """
        :param distr: The target distribution of audio tags that we want to
            match. If this is not supplied, the function simply checks that
            there are some audio files.
        :parma distr: dict
        :return: True if there are audio files.
        :rtype: boolean
        """
        if distr is None:
            return len(self.id_set) > 0
        else:
            for tag in distr:
                if tag not in self.duration_to_list_per_tag:
                    return False
            return True

    def _load_all_records_from_disk(self, audio_dir, idx_fname, bin_size):
        """Loads all audio records from the disk into memory and groups them
        into chunks based on their duration and the bin_size granalarity.

        Once all the records are read, indices are built from these records
        by another function so that the audio samples can be drawn efficiently.

        Updates:
            self.audio_dir (path): audio root directory
            self.idx_fname (path): audio database index filename
            self.bin_size (float): granularity of bins
            self.idx_to_record (dict): maps from the audio id to
                (duration, file_size, relative_path, tags)
            self.tag_to_id_set (dict): maps from the tag to
                the set of id's of audios that have this tag.
            self.id_set_complete (set): set of all audio id's in the index file
            self.min_duration (float): minimum audio duration observed in the
                index file
            self.duration_bins (list): the lower bounds on the duration of
                audio files falling in each bin
            self.duration_to_id_set (dict): contains (k, v) where v is the set
                of id's of audios whose lengths are longer than or equal to k.
                (e.g. k is the duration lower bound of this bin).
            self.duration_to_id_set_per_tag (dict): Something like above but
                has a finer granularity mapping from the tag to
                duration_to_id_set.
            self.shared_duration_bins (list): list of sets where each set
                contains duration lower bounds whose audio id sets are the
                same. The rationale for having this is that there are a few
                but extremely long audio files which lead to a lot of bins.
                When the id sets do not change across various minimum duration
                boundaries, we
                cluster these together and make them point to the same id set
                reference.

        :return: whether the records were read from the disk. The assumption is
            that the audio index file on disk and the actual audio files
            are constructed once and never change during training. We only
            re-read when either the directory or the index file path change.
        """
        if self.audio_dir == audio_dir and self.idx_fname == idx_fname and \
           self.bin_size == bin_size:
            # The audio directory and/or the list of audio files
            # haven't changed. No need to load the list again.
            return False

        # Remember where the audio index is most recently read from.
        self.audio_dir = audio_dir
        self.idx_fname = idx_fname
        self.bin_size = bin_size

        # Read in the idx and compute the number of bins necessary
        self.clear()
        rank = []
        min_duration = float('inf')
        max_duration = float('-inf')
        for idx, duration, file_size, relpath, tags in \
                stream_audio_index(idx_fname):
            self.idx_to_record[idx] = (duration, file_size, relpath, tags)
            max_duration = max(max_duration, duration)
            min_duration = min(min_duration, duration)
            rank.append((duration, idx))
            for tag in tags:
                self.tag_to_id_set[tag].add(idx)
        if len(rank) == 0:
            # file is empty
            raise IOError("Index file {} is empty".format(idx_fname))
        for tag in self.tag_to_id_set:
            self.id_set_complete |= self.tag_to_id_set[tag]
        dur = min_duration
        self.min_duration = min_duration
        while dur < max_duration + bin_size:
            self.duration_bins.append(dur)
            dur += bin_size

        # Sort in decreasing order of duration and populate
        # the cumulative indices lists.
        rank.sort(reverse=True)

        # These are indices for `rank` and used to keep track of whether
        # there are new records to add in the current bin.
        last = 0
        cur = 0

        # The set of audios falling in the previous bin; in the case,
        # where we don't find new audios for the current bin, we store
        # the reference to the last set so as to conserve memory.
        # This is not such a big problem if the audio duration is
        # bounded by a small number like 30 seconds and the
        # bin size is big enough. But, for raw freesound audios,
        # some audios can be as long as a few hours!
        last_audio_set = set()

        # The same but for each tag so that we can pick audios based on
        # tags and also some user-specified tag distribution.
        last_audio_set_per_tag = defaultdict(lambda: set())

        # Set of lists of bins sharing the same audio sets.
        shared = set()

        for i in range(len(self.duration_bins) - 1, -1, -1):
            lower_bound = self.duration_bins[i]
            new_audio_idxs = set()
            new_audio_idxs_per_tag = defaultdict(lambda: set())
            while cur < len(rank) and rank[cur][0] >= lower_bound:
                idx = rank[cur][1]
                tags = self.idx_to_record[idx][3]
                new_audio_idxs.add(idx)
                for tag in tags:
                    new_audio_idxs_per_tag[tag].add(idx)
                cur += 1
            # This makes certain that the same list is shared across
            # different bins if no new indices are added.
            if cur == last:
                shared.add(lower_bound)
            else:
                last_audio_set = last_audio_set | new_audio_idxs
                for tag in new_audio_idxs_per_tag:
                    last_audio_set_per_tag[tag] = \
                        last_audio_set_per_tag[tag] | \
                        new_audio_idxs_per_tag[tag]
                if len(shared) > 0:
                    self.shared_duration_bins.append(shared)
                shared = set([lower_bound])
                ### last_audio_set = set()  should set blank
            last = cur
            self.duration_to_id_set[lower_bound] = last_audio_set
            for tag in last_audio_set_per_tag:
                self.duration_to_id_set_per_tag[lower_bound][tag] = \
                    last_audio_set_per_tag[tag]

        # The last `shared` record isn't added to the `shared_duration_bins`.
        self.shared_duration_bins.append(shared)

        # We make sure that the while loop above has exhausted through the
        # `rank` list by checking if the `cur`rent index in `rank` equals
        # the length of the array, which is the halting condition.
        assert cur == len(rank)

        return True

    def _build_index_from_records(self, tag_list):
        """ Uses the in-memory records read from the index file to build
        an in-memory index restricted to the given tag list.

        :param tag_list: List of tags we are interested in sampling from.
        :type tag_list: list(str)

        Updates:
            self.id_set (set): the set of all audio id's that can be sampled.
            self.duration_to_list (dict): maps from the duration lower bound
                to the id's of audios longer than this duration.
            self.duration_to_list_per_tag (dict): maps from the tag to
                the same structure as self.duration_to_list. This is to support
                sampling from a target noise distribution.

        :return: whether the index was built from scratch
        """
        if self.tags == tag_list:
            return False

        self.tags = tag_list
        if len(tag_list) == 0:
            self.id_set = self.id_set_complete
        else:
            self.id_set = set()
            for tag in tag_list:
                self.id_set |= self.tag_to_id_set[tag]

        # Next, we need to take a subset of the audio files
        for shared in self.shared_duration_bins:
            # All bins in `shared' have the same index lists
            # so we can intersect once and set all of them to this list.
            lb = list(shared)[0]
            intersected = list(self.id_set & self.duration_to_id_set[lb])
            duration_to_id_set = self.duration_to_id_set_per_tag[lb]
            intersected_per_tag = {
                tag: self.tag_to_id_set[tag] & duration_to_id_set[tag]
                for tag in duration_to_id_set
            }
            for bin_key in shared:
                self.duration_to_list[bin_key] = intersected
                for tag in intersected_per_tag:
                    self.duration_to_list_per_tag[tag][bin_key] = \
                        intersected_per_tag[tag]
        assert len(self.duration_to_list) == len(self.duration_to_id_set)
        return True

    def refresh_records_from_index_file(self,
                                        audio_dir,
                                        idx_fname,
                                        tag_list,
                                        bin_size=2.0):
        """ Loads the index file and populates the records
        for building the internal index.

        If the audio directory or index file name has changed, the whole index
        is reloaded from scratch. If only the tag_list is changed, then the
        desired index is built from the complete, in-memory record.

        :param audio_dir: audio directory
        :type audio_dir: basestring
        :param idx_fname: audio index file name
        :type idex_fname: basestring
        :param tag_list: list of tags we are interested in loading;
            if empty, we load all.
        :type tag_list: list
        :param bin_size: optional argument for controlling the granularity
            of duration bins
        :type bin_size: float
        """
        if tag_list is None:
            tag_list = []
        reloaded_records = self._load_all_records_from_disk(audio_dir,
                                                            idx_fname, bin_size)
        if reloaded_records or self.tags != tag_list:
            self._build_index_from_records(tag_list)
            logger.info('loaded {} audio files from {}'
                        .format(len(self.id_set), idx_fname))

    def sample_audio(self, duration, rng=None, distr=None):
        """ Uniformly draws an audio record of at least the desired duration

        :param duration: minimum desired audio duration
        :type duration: float
        :param rng: random number generator
        :type rng: random.Random
        :param distr: target distribution of audio tags. If not provided,
        :type distr: dict
        all audio files are sampled uniformly at random.

        :returns: success, (duration, file_size, path)
        """
        if duration < 0.0:
            duration = self.min_duration
        i = bisect.bisect_left(self.duration_bins, duration)
        if i == len(self.duration_bins):
            return False, None
        bin_key = self.duration_bins[i]
        if distr is None:
            indices = self.duration_to_list[bin_key]
        else:
            # If a desired audio distribution is given, we sample from it.
            if rng is None:
                rng = random.Random()
            nprng = np.random.RandomState(rng.getrandbits(32))
            prob_masses = distr.values()
            prob_masses /= np.sum(prob_masses)
            tag = nprng.choice(distr.keys(), p=prob_masses)
            indices = self.duration_to_list_per_tag[tag][bin_key]
        if len(indices) == 0:
            return False, None
        else:
            if rng is None:
                rng = random.Random()
            # duration, file size and relative path from root
            s = self.idx_to_record[rng.sample(indices, 1)[0]]
            s = (s[0], s[1], os.path.join(self.audio_dir, s[2]))
            return True, s
