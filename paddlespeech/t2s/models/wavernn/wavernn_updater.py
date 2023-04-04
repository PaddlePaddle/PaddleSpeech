# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pathlib import Path

import paddle
import soundfile as sf
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer
from paddle.optimizer import Optimizer

from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_grad_norm(parameters, norm_type: str = 2):
    '''
    calculate grad norm of mdoel's parameters
    parameters:
        model's parameters
    norm_type: str
    Returns
    ------------
    Tensor
        grad_norm
    '''

    grad_list = [
        paddle.to_tensor(p.grad) for p in parameters if p.grad is not None
    ]
    norm_list = paddle.stack(
        [paddle.norm(grad, norm_type) for grad in grad_list])
    total_norm = paddle.norm(norm_list)
    return total_norm


# for save name in gen_valid_samples()
ITERATION = 0


class WaveRNNUpdater(StandardUpdater):
    def __init__(self,
                 model: Layer,
                 optimizer: Optimizer,
                 criterion: Layer,
                 dataloader: DataLoader,
                 init_state=None,
                 output_dir: Path = None,
                 mode='RAW'):
        super().__init__(model, optimizer, dataloader, init_state=None)

        self.criterion = criterion
        # self.scheduler = scheduler

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""
        self.mode = mode

    def update_core(self, batch):

        self.msg = "Rank: {}, ".format(dist.get_rank())
        losses_dict = {}
        # parse batch
        self.model.train()
        self.optimizer.clear_grad()

        wav, y, mel = batch

        y_hat = self.model(wav, mel)
        if self.mode == 'RAW':
            y_hat = y_hat.transpose([0, 2, 1]).unsqueeze(-1)
        elif self.mode == 'MOL':
            y_hat = paddle.cast(y, dtype='float32')

        y = y.unsqueeze(-1)
        loss = self.criterion(y_hat, y)
        loss.backward()
        grad_norm = float(
            calculate_grad_norm(self.model.parameters(), norm_type=2))

        self.optimizer.step()

        report("train/loss", float(loss))
        report("train/grad_norm", float(grad_norm))

        losses_dict["loss"] = float(loss)
        losses_dict["grad_norm"] = float(grad_norm)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        global ITERATION
        ITERATION = self.state.iteration + 1


class WaveRNNEvaluator(StandardEvaluator):
    def __init__(self,
                 model: Layer,
                 criterion: Layer,
                 dataloader: Optimizer,
                 output_dir: Path = None,
                 valid_generate_loader=None,
                 config=None):
        super().__init__(model, dataloader)

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

        self.criterion = criterion
        self.valid_generate_loader = valid_generate_loader
        self.config = config
        self.mode = config.model.mode

        self.valid_samples_dir = output_dir / "valid_samples"
        self.valid_samples_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_core(self, batch):
        self.msg = "Evaluate: "
        losses_dict = {}
        # parse batch
        wav, y, mel = batch
        y_hat = self.model(wav, mel)

        if self.mode == 'RAW':
            y_hat = y_hat.transpose([0, 2, 1]).unsqueeze(-1)
        elif self.mode == 'MOL':
            y_hat = paddle.cast(y, dtype='float32')

        y = y.unsqueeze(-1)
        loss = self.criterion(y_hat, y)
        report("eval/loss", float(loss))

        losses_dict["loss"] = float(loss)

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)

    def gen_valid_samples(self):

        for i, item in enumerate(self.valid_generate_loader):
            if i >= self.config.generate_num:
                break
            print('\n| Generating: {}/{}'.format(i + 1,
                                                 self.config.generate_num))

            mel = item['feats']
            wav = item['wave']
            wav = wav.squeeze(0)

            origin_save_path = self.valid_samples_dir / '{}_steps_{}_target.wav'.format(
                self.iteration, i)
            sf.write(origin_save_path, wav.numpy(), samplerate=self.config.fs)

            if self.config.inference.gen_batched:
                batch_str = 'gen_batched_target{}_overlap{}'.format(
                    self.config.inference.target, self.config.inference.overlap)
            else:
                batch_str = 'gen_not_batched'
            gen_save_path = str(
                self.valid_samples_dir /
                '{}_steps_{}_{}.wav'.format(self.iteration, i, batch_str))
            # (1, T, C_aux) -> (T, C_aux)
            mel = mel.squeeze(0)
            gen_sample = self.model.generate(mel,
                                             self.config.inference.gen_batched,
                                             self.config.inference.target,
                                             self.config.inference.overlap,
                                             self.config.mu_law)
            sf.write(gen_save_path,
                     gen_sample.numpy(),
                     samplerate=self.config.fs)

    def __call__(self, trainer=None):
        summary = self.evaluate()
        for k, v in summary.items():
            report(k, v)
        # gen samples at then end of evaluate
        self.iteration = ITERATION
        if self.iteration % self.config.gen_eval_samples_interval_steps == 0:
            self.gen_valid_samples()
