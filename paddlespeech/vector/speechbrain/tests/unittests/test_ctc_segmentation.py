from speechbrain.pretrained import EncoderDecoderASR
import pytest

pytest.importorskip(
    "speechbrain.alignment.ctc_segmentation",
    reason="These tests require the ctc_segmentation library",
)


@pytest.fixture()
def asr_model():
    """Load model for the CTC segmentation test."""

    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech"
    )
    return asr_model


def test_CTCSegmentation(asr_model: EncoderDecoderASR):
    """Test CTC segmentation.

    Instead of pre-loading an ASR model and inferring an audio file, it is also
    possible to use randomly generated ASR models and speech data. Please note
    that with random data, there will be a small chance that this test might
    randomly fail.
    """

    import numpy as np
    from speechbrain.alignment.ctc_segmentation import CTCSegmentation
    from speechbrain.alignment.ctc_segmentation import CTCSegmentationTask

    # speech either from the test audio file or random
    # example file included in the speechbrain repository
    # speech = "./samples/audio_samples/example1.wav"
    num_samples = 100000
    speech = np.random.randn(num_samples)

    # text includes:
    #   one blank line
    #   kaldi-style utterance names
    #   one char not included in char list
    text = (
        "\n"
        "utt_a THE BIRCH CANOE\n"
        "utt_b SLID ON THE\n"
        "utt_c SMOOTH PLANKS\n"
    )
    aligner = CTCSegmentation(
        asr_model=asr_model, kaldi_style_text=True, min_window_size=10,
    )
    segments = aligner(speech, text)
    # check segments
    assert isinstance(segments, CTCSegmentationTask)
    kaldi_text = str(segments)
    first_line = kaldi_text.splitlines()[0]
    assert "utt_a" == first_line.split(" ")[0]
    start, end, score = segments.segments[0]
    assert start > 0.0
    assert end >= start
    assert score < 0.0
    # check options and align with "classic" text converter
    option_dict = {
        "time_stamps": "fixed",
        "samples_to_frames_ratio": 512,
        "min_window_size": 100,
        "max_window_size": 20000,
        "set_blank": 0,
        "scoring_length": 10,
        "replace_spaces_with_blanks": True,
        "gratis_blank": True,
        "kaldi_style_text": False,
        "text_converter": "classic",
    }
    aligner.set_config(**option_dict)
    assert aligner.warned_about_misconfiguration
    text = [
        "THE LITTLE GIRL",
        "HAD BEEN ASLEEP",
        "BUT SHE HEARD THE RAPS",
        "AND OPENED THE DOOR",
    ]
    segments = aligner(speech, text, name="foo")
    segments_str = str(segments)
    first_line = segments_str.splitlines()[0]
    assert "foo_0000" == first_line.split(" ")[0]
    # test the ratio estimation (result: 509)
    ratio = aligner.estimate_samples_to_frames_ratio()
    assert 400 <= ratio <= 700
