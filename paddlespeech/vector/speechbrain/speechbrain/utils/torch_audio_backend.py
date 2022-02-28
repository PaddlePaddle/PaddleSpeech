import platform
import logging


logger = logging.getLogger(__name__)


def check_torchaudio_backend():
    """Checks the torchaudio backend and sets it to soundfile if
    windows is detected.
    """
    current_system = platform.system()
    if current_system == "Windows":
        logger.warn(
            "The torchaudio backend is switched to 'soundfile'. Note that 'sox_io' is not supported on Windows."
        )
        torchaudio.set_audio_backend("soundfile")
