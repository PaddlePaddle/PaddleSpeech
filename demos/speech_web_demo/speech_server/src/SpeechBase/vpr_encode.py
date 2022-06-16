from paddlespeech.cli import VectorExecutor
import numpy as np
import logging

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

if __name__ == '__main__':
    audio_path = r"../../source/demo/demo_16k.wav"
    emb = get_audio_embedding(audio_path)
    print(emb.shape)
    print(emb.dtype)
    print(type(emb))
    