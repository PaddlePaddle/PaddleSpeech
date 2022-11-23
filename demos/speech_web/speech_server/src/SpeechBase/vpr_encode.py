import logging

import numpy as np

from paddlespeech.cli.vector import VectorExecutor

vector_executor = VectorExecutor()


def get_audio_embedding(path):
    """
    Use vpr_inference to generate embedding of audio
    """
    try:
        embedding = vector_executor(
            audio_file=path, model='ecapatdnn_voxceleb12')
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        logging.error(f"Error with embedding:{e}")
        return None
