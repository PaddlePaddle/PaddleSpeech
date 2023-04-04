import numpy as np

from paddlespeech.server.engine.asr.online.python.asr_engine import ASREngine
from paddlespeech.server.engine.asr.online.python.asr_engine import PaddleASRConnectionHanddler
from paddlespeech.server.utils.config import get_config


def readWave(samples):
    x_len = len(samples)

    chunk_size = 85 * 16  #80ms, sample_rate = 16kHz
    if x_len % chunk_size != 0:
        padding_len_x = chunk_size - x_len % chunk_size
    else:
        padding_len_x = 0

    padding = np.zeros((padding_len_x), dtype=samples.dtype)
    padded_x = np.concatenate([samples, padding], axis=0)

    assert (x_len + padding_len_x) % chunk_size == 0
    num_chunk = (x_len + padding_len_x) / chunk_size
    num_chunk = int(num_chunk)
    for i in range(0, num_chunk):
        start = i * chunk_size
        end = start + chunk_size
        x_chunk = padded_x[start:end]
        yield x_chunk


class ASR:
    def __init__(
        self,
        config_path,
    ) -> None:
        self.config = get_config(config_path)['asr_online']
        self.engine = ASREngine()
        self.engine.init(self.config)
        self.connection_handler = PaddleASRConnectionHanddler(self.engine)

    def offlineASR(self, samples, sample_rate=16000):
        x_chunk, x_chunk_lens = self.engine.preprocess(samples=samples,
                                                       sample_rate=sample_rate)
        self.engine.run(x_chunk, x_chunk_lens)
        result = self.engine.postprocess()
        self.engine.reset()
        return result

    def onlineASR(self, samples: bytes = None, is_finished=False):
        if not is_finished:
            # 流式开始
            self.connection_handler.extract_feat(samples)
            self.connection_handler.decode(is_finished)
            asr_results = self.connection_handler.get_result()
            return asr_results
        else:
            # 流式结束
            self.connection_handler.decode(is_finished=True)
            self.connection_handler.rescoring()
            asr_results = self.connection_handler.get_result()
            self.connection_handler.reset()
            return asr_results
