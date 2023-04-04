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
from contextlib import nullcontext

import paddle
from paddle import distributed as dist

from paddlespeech.s2t.training.extensions.evaluator import StandardEvaluator
from paddlespeech.s2t.training.reporter import report
from paddlespeech.s2t.training.timer import Timer
from paddlespeech.s2t.training.updaters.standard_updater import StandardUpdater
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


class U2Evaluator(StandardEvaluator):
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)
        self.msg = ""
        self.num_seen_utts = 0
        self.total_loss = 0.0

    def evaluate_core(self, batch):
        self.msg = "Valid: Rank: {}, ".format(dist.get_rank())
        losses_dict = {}

        loss, attention_loss, ctc_loss = self.model(*batch[1:])
        if paddle.isfinite(loss):
            num_utts = batch[1].shape[0]
            self.num_seen_utts += num_utts
            self.total_loss += float(loss) * num_utts

            losses_dict['loss'] = float(loss)
            if attention_loss:
                losses_dict['att_loss'] = float(attention_loss)
            if ctc_loss:
                losses_dict['ctc_loss'] = float(ctc_loss)

            for k, v in losses_dict.items():
                report("eval/" + k, v)

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        logger.info(self.msg)
        return self.total_loss, self.num_seen_utts


class U2Updater(StandardUpdater):
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 dataloader,
                 init_state=None,
                 accum_grad=1,
                 **kwargs):
        super().__init__(model,
                         optimizer,
                         scheduler,
                         dataloader,
                         init_state=init_state)
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.msg = ""

    def update_core(self, batch):
        """One Step

        Args:
            batch (List[Object]): utts, xs, xlens, ys, ylens
        """
        losses_dict = {}
        self.msg = "Rank: {}, ".format(dist.get_rank())

        # forward
        batch_size = batch[1].shape[0]
        loss, attention_loss, ctc_loss = self.model(*batch[1:])
        # loss div by `batch_size * accum_grad`
        loss /= self.accum_grad

        # loss backward
        if (self.forward_count + 1) != self.accum_grad:
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            context = self.model.no_sync
        else:
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            context = nullcontext

        with context():
            loss.backward()
            layer_tools.print_grads(self.model, print_func=None)

        # loss info
        losses_dict['loss'] = float(loss) * self.accum_grad
        if attention_loss:
            losses_dict['att_loss'] = float(attention_loss)
        if ctc_loss:
            losses_dict['ctc_loss'] = float(ctc_loss)
        # report loss
        for k, v in losses_dict.items():
            report("train/" + k, v)
        # loss msg
        self.msg += "batch size: {}, ".format(batch_size)
        self.msg += "accum: {}, ".format(self.accum_grad)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())

        # Truncate the graph
        loss.detach()

        # update parameters
        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0

        self.optimizer.step()
        self.optimizer.clear_grad()
        self.scheduler.step()

    def update(self):
        # model is default in train mode

        # training for a step is implemented here
        with Timer("data time cost:{}"):
            batch = self.read_batch()
        with Timer("step time cost:{}"):
            self.update_core(batch)

        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        if self.forward_count == 0:
            self.state.iteration += 1
        if self.updates_per_epoch is not None:
            if self.state.iteration % self.updates_per_epoch == 0:
                self.state.epoch += 1
