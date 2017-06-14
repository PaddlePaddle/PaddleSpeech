""" noise speech
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import os
from collections import defaultdict

from . import base
from . import audio_database
from data_utils.speech import SpeechSegment

TURK = "turk"
USE_AUDIO_DATABASE_SOURCES = frozenset(["freesound", "chime"])
HALF_NOISE_LENGTH_MIN_THRESHOLD = 3.0
FIND_NOISE_MAX_ATTEMPTS = 20

logger = logging.getLogger(__name__)


def get_first_smaller(items, value):
    index = bisect.bisect_left(items, value) - 1
    assert items[index] < value, \
        'get_first_smaller failed! %d %d' % (items[index], value)
    return items[index]


def get_first_larger(items, value):
    'Find leftmost value greater than value'
    index = bisect.bisect_right(items, value)
    assert index < len(items), \
        "no noise bin exists for this audio length (%f)" % value
    assert items[index] > value, \
        'get_first_larger failed! %d %d' % (items[index], value)
    return items[index]


def _get_turk_noise_files(noise_dir, index_file):
    """ Creates a map from duration => a list of noise filenames

    :param noise_dir: Directory of noise files which contains
        "noise-samples-list"
    :type noise_dir: basestring
    :param index_file: Noise list
    :type index_file: basestring

    returns:noise_files (defaultdict): A map of bins to noise files.
        Each key is the duration, and the value is a list of noise
        files binned to this duration. Each bin is 2 secs.

    Note: noise-samples-list should contain one line per noise (wav) file
        along with its duration in milliseconds
    """
    noise_files = defaultdict(list)
    if not os.path.exists(index_file):
        logger.error('No noise files were found at {}'.format(index_file))
        return noise_files
    num_noise_files = 0
    rounded_durations = list(range(0, 65, 2))
    with open(index_file, 'r') as fl:
        for line in fl:
            fname = os.path.join(noise_dir, line.strip().split()[0])
            duration = float(line.strip().split()[1]) / 1000
            # bin the noise files into length bins rounded by 2 sec
            bin_id = get_first_smaller(rounded_durations, duration)
            noise_files[bin_id].append(fname)
            num_noise_files += 1
    logger.info('Loaded {} turk noise files'.format(num_noise_files))
    return noise_files


class NoiseSpeechAugmentor(base.AugmentorBase):
    """ Noise addition block

    :param snr_min: minimum signal-to-noise ratio
    :type snr_min: float
    :param snr_max: maximum signal-to-noise ratio
    :type snr_max: float
    :param noise_dir: root of where noise files are stored
    :type noise_fir: basestring
    :param index_file: index of noises of interest in noise_dir
    :type index_file: basestring
    :param source: select one from
        - turk
        - freesound
        - chime
        Note that this field is no longer required for the freesound
        and chime
    :type source: string
    :param tags: optional parameter for specifying what
        particular noises we want to add. See above for the available tags.
    :type tags: list
    :param tag_distr: optional noise distribution
    :type tag_distr: dict
    """

    def __init__(self,
                 rng,
                 snr_min,
                 snr_max,
                 noise_dir,
                 source,
                 allow_downsampling=None,
                 index_file=None,
                 tags=None,
                 tag_distr=None):
        # Define all required parameter maps here.
        self.rng = rng
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_dir = noise_dir
        self.source = source

        self.allow_downsampling = allow_downsampling
        self.index_file = index_file
        self.tags = tags
        self.tag_distr = tag_distr

        # When new noise sources are added, make sure to define the
        # associated bookkeeping variables here.
        self.turk_noise_files = []
        self.turk_noise_dir = None
        self.audio_index = audio_database.AudioIndex()

    def _init_data(self):
        """ Preloads stuff from disk in an attempt (e.g. list of files, etc)
        to make later loading faster. If the data configuration remains the
        same, this function does nothing.

        """
        noise_dir = self.noise_dir
        index_file = self.index_file
        source = self.source
        if not index_file:
            if source == TURK:
                index_file = os.path.join(noise_dir, 'noise-samples-list')
                logger.debug("index_file not provided; " + "defaulting to " +
                             index_file)
            else:
                if source != "":
                    assert source in USE_AUDIO_DATABASE_SOURCES, \
                        "{} not supported by audio_database".format(source)
                index_file = os.path.join(noise_dir,
                                          "audio_index_commercial.txt")
                logger.debug("index_file not provided; " + "defaulting to " +
                             index_file)

        if source == TURK:
            if self.turk_noise_dir != noise_dir:
                self.turk_noise_dir = noise_dir
                self.turk_noise_files = _get_turk_noise_files(noise_dir,
                                                              index_file)
        # elif source == TODO_SUPPORT_NON_AUDIO_DATABASE_BASED_SOURCES:
        else:
            if source != "":
                assert source in USE_AUDIO_DATABASE_SOURCES, \
                    "{} not supported by audio_database".format(source)
            self.audio_index.refresh_records_from_index_file(
                self.noise_dir, index_file, self.tags)

    def transform_audio(self, audio_segment):
        """Adds walla noise

        :param audio_segment: Input audio
        :type audio_segment: SpeechSegment
        """
        # This handles the cases where the data source or directories change.
        self._init_data
        source = self.source
        allow_downsampling = self.allow_downsampling
        if source == TURK:
            self._add_turk_noise(audio_segment, self.rng, allow_downsampling)
        # elif source == TODO_SUPPORT_NON_AUDIO_DATABASE_BASED_SOURCES:
        else:
            self._add_noise(audio_segment, self.rng, allow_downsampling)

    def _sample_snr(self):
        """ Returns a float sampled in [`self.snr_min`, `self.snr_max`]
        if both `self.snr_min` and `self.snr_max` are non-zero.
        """
        snr_min = self.snr_min
        snr_max = self.snr_max
        sampled_snr = self.rng.uniform(snr_min, snr_max)
        return sampled_snr

    def _add_turk_noise(self, audio_segment, allow_downsampling):
        """ Adds a turk noise to the input audio.

        :param audio_segment: input audio
        :type audio_segment: audiosegment
        :param allow_downsampling: indicates whether downsampling
            is allowed
        :type allow_downsampling: boolean 
        """
        read_size = 0
        if len(self.turk_noise_files) > 0:
            snr = self._sample_snr(self.rng)
            # Draw the noise file randomly from noise files that are
            # slightly longer than the utterance
            noise_bins = sorted(self.turk_noise_files.keys())
            # note some bins can be empty, so we can't just round up
            # to the nearest 2-sec interval
            rounded_duration = get_first_larger(noise_bins,
                                                audio_segment.duration)
            noise_fname = \
                self.rng.sample(self.turk_noise_files[rounded_duration], 1)[0]
            noise = SpeechSegment.from_wav_file(noise_fname)
            logger.debug('noise_fname {}'.format(noise_fname))
            logger.debug('snr {}'.format(snr))
            read_size = len(noise) * 2
            # May throw exceptions, but this is caught by
            # AudioFeaturizer.get_audio_files.
            audio_segment.add_noise(
                noise, snr, rng=self.rng, allow_downsampling=allow_downsampling)

    def _add_noise(self, audio_segment, allow_downsampling):
        """ Adds a noise indexed in audio_database.AudioIndex.

        :param audio_segment: input audio
        :type audio_segment: SpeechSegment
        :param allow_downsampling: indicates whether downsampling
            is allowed
        :type allow_downsampling: boolean

        Returns:
            (SpeechSegment, int)
                - sound with turk noise added
                - number of bytes read from disk
        """
        read_size = 0
        tag_distr = self.tag_distr
        if not self.audio_index.has_audio(tag_distr):
            if tag_distr is None:
                if not self.tags:
                    raise RuntimeError("The noise index does not have audio "
                                       "files to sample from.")
                else:
                    raise RuntimeError("The noise index does not have audio "
                                       "files of the given tags to sample "
                                       "from.")
            else:
                raise RuntimeError("The noise index does not have audio "
                                   "files to match the target noise "
                                   "distribution.")
        else:
            # Compute audio segment related statistics
            audio_duration = audio_segment.duration

            # Sample relevant augmentation parameters.
            snr = self._sample_snr(self.rng)

            # Perhaps, we may not have a sufficiently long noise, so we need
            # to search iteratively.
            min_duration = audio_duration + 0.25
            for _ in range(FIND_NOISE_MAX_ATTEMPTS):
                logger.debug("attempting to find noise of length "
                             "at least {}".format(min_duration))

                success, record = \
                    self.audio_index.sample_audio(min_duration,
                                                  rng=self.rng,
                                                  distr=tag_distr)

                if success is True:
                    noise_duration, read_size, noise_fname = record

                    # Assert after logging so we know
                    # what caused augmentation to fail.
                    logger.debug("noise_fname {}".format(noise_fname))
                    logger.debug("snr {}".format(snr))
                    assert noise_duration >= min_duration
                    break

                # Decrease the desired minimum duration linearly.
                # If the value becomes smaller than some threshold,
                # we half the value instead.
                if min_duration > HALF_NOISE_LENGTH_MIN_THRESHOLD:
                    min_duration -= 2.0
                else:
                    min_duration *= 0.5

            if success is False:
                logger.info("Failed to find a noise file")
                return

            diff_duration = audio_duration + 0.25 - noise_duration
            if diff_duration >= 0.0:
                # Here, the noise is shorter than the audio file, so
                # we pad with zeros to make sure the noise sound is applied
                # with a uniformly random shift.
                noise = SpeechSegment.from_file(noise_fname)
                noise = noise.pad_silence(diff_duration, sides="both")
            else:
                # The noise clip is at least ~25 ms longer than the audio
                # segment here.
                diff_duration = int(noise_duration * audio_segment.sample_rate) - \
                    int(audio_duration * audio_segment.sample_rate) - \
                    int(0.02 * audio_segment.sample_rate)
                start = float(self.rng.randint(0, diff_duration)) / \
                    audio.sample_rate
                finish = min(start + audio_duration + 0.2, noise_duration)
                noise = SpeechSegment.slice_from_file(noise_fname, start,
                                                      finish)

            if len(noise) < len(audio_segment):
                # This is to ensure that the noise clip is at least as
                # long as the audio segment.
                num_samples_to_pad = len(audio_segment) - len(noise)
                # Padding this amount of silence on both ends ensures that
                # the placement of the noise clip is uniformly random.
                silence = SpeechSegment(
                    np.zeros(num_samples_to_pad), audio_segment.sample_rate)
                noise = SpeechSegment.concatenate(silence, noise, silence)

            audio_segment.add_noise(
                noise, snr, rng=self.rng, allow_downsampling=allow_downsampling)
