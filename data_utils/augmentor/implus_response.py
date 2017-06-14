""" Impulse response"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base
from . import audio_database
from data_utils.speech import SpeechSegment


class ImpulseResponseAugmentor(base.AugmentorBase):
    """ Instantiates an impulse response model

    :param ir_dir: directory containing impulse responses
    :type ir_dir: basestring
    :param tags: optional parameter for specifying what
            particular impulse responses to apply.
    :type tags: list
    :parm tag_distr: optional noise distribution
    :type tag_distr: dict
    """

    def __init__(self, rng, ir_dir, index_file, tags=None, tag_distr=None):
        # Define all required parameter maps here.
        self.ir_dir = ir_dir
        self.index_file = index_file

        self.tags = tags
        self.tag_distr = tag_distr

        self.audio_index = audio_database.AudioIndex()
        self.rng = rng

    def _init_data(self):
        """ Preloads stuff from disk in an attempt (e.g. list of files, etc)
        to make later loading faster. If the data configuration remains the
        same, this function does nothing.

        """
        self.audio_index.refresh_records_from_index_file(
            self.ir_dir, self.index_file, self.tags)

    def transform_audio(self, audio_segment):
        """ Convolves the input audio with an impulse response.

        :param audio_segment: input audio
        :type audio_segment: AudioSegemnt
        """
        # This handles the cases where the data source or directories change.
        self._init_data()

        read_size = 0
        tag_distr = self.tag_distr
        if not self.audio_index.has_audio(tag_distr):
            if tag_distr is None:
                if not self.tags:
                    raise RuntimeError("The ir index does not have audio "
                                       "files to sample from.")
                else:
                    raise RuntimeError("The ir index does not have audio "
                                       "files of the given tags to sample "
                                       "from.")
            else:
                raise RuntimeError("The ir index does not have audio "
                                   "files to match the target ir "
                                   "distribution.")
        else:
            # Querying with a negative duration triggers the index to search
            # from all impulse responses.
            success, record = self.audio_index.sample_audio(
                -1.0, rng=self.rng, distr=tag_distr)
            if success is True:
                _, read_size, ir_fname = record
                ir_wav = SpeechSegment.from_file(ir_fname)
                audio_segment.convolve(ir_wav, allow_resampling=True)
