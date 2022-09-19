from .sox_effects import apply_effects_file
from .sox_effects import apply_effects_tensor
from .sox_effects import effect_names
from .sox_effects import init_sox_effects
from .sox_effects import shutdown_sox_effects
from paddlespeech.audio._internal import module_utils as _mod_utils

if _mod_utils.is_sox_available():
    import atexit

    init_sox_effects()
    atexit.register(shutdown_sox_effects)

__all__ = [
    "init_sox_effects",
    "shutdown_sox_effects",
    "effect_names",
    "apply_effects_tensor",
    "apply_effects_file",
]
