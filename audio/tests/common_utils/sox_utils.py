import subprocess
import sys
import warnings


def get_encoding(dtype):
    encodings = {
        "float32": "floating-point",
        "int32": "signed-integer",
        "int16": "signed-integer",
        "uint8": "unsigned-integer",
    }
    return encodings[dtype]


def get_bit_depth(dtype):
    bit_depths = {
        "float32": 32,
        "int32": 32,
        "int16": 16,
        "uint8": 8,
    }
    return bit_depths[dtype]


def gen_audio_file(
    path,
    sample_rate,
    num_channels,
    *,
    encoding=None,
    bit_depth=None,
    compression=None,
    attenuation=None,
    duration=1,
    comment_file=None,
):
    """Generate synthetic audio file with `sox` command."""
    if path.endswith(".wav"):
        warnings.warn(
            "Use get_wav_data and save_wav to generate wav file for accurate result."
        )
    command = [
        "sox",
        "-V3",  # verbose
        "--no-dither",  # disable automatic dithering
        "-R",
        # -R is supposed to be repeatable, though the implementation looks suspicious
        # and not setting the seed to a fixed value.
        # https://fossies.org/dox/sox-14.4.2/sox_8c_source.html
        # search "sox_globals.repeatable"
    ]
    if bit_depth is not None:
        command += ["--bits", str(bit_depth)]
    command += [
        "--rate",
        str(sample_rate),
        "--null",  # no input
        "--channels",
        str(num_channels),
    ]
    if compression is not None:
        command += ["--compression", str(compression)]
    if bit_depth is not None:
        command += ["--bits", str(bit_depth)]
    if encoding is not None:
        command += ["--encoding", str(encoding)]
    if comment_file is not None:
        command += ["--comment-file", str(comment_file)]
    command += [
        str(path),
        "synth",
        str(duration),  # synthesizes for the given duration [sec]
        "sawtooth",
        "1",
        # saw tooth covers the both ends of value range, which is a good property for test.
        # similar to linspace(-1., 1.)
        # this introduces bigger boundary effect than sine when converted to mp3
    ]
    if attenuation is not None:
        command += ["vol", f"-{attenuation}dB"]
    print(" ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)


def convert_audio_file(src_path,
                       dst_path,
                       *,
                       encoding=None,
                       bit_depth=None,
                       compression=None):
    """Convert audio file with `sox` command."""
    command = ["sox", "-V3", "--no-dither", "-R", str(src_path)]
    if encoding is not None:
        command += ["--encoding", str(encoding)]
    if bit_depth is not None:
        command += ["--bits", str(bit_depth)]
    if compression is not None:
        command += ["--compression", str(compression)]
    command += [dst_path]
    print(" ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)


def _flattern(effects):
    if not effects:
        return effects
    if isinstance(effects[0], str):
        return effects
    return [item for sublist in effects for item in sublist]


def run_sox_effect(input_file,
                   output_file,
                   effect,
                   *,
                   output_sample_rate=None,
                   output_bitdepth=None):
    """Run sox effects"""
    effect = _flattern(effect)
    command = ["sox", "-V", "--no-dither", input_file]
    if output_bitdepth:
        command += ["--bits", str(output_bitdepth)]
    command += [output_file] + effect
    if output_sample_rate:
        command += ["rate", str(output_sample_rate)]
    print(" ".join(command))
    subprocess.run(command, check=True)
