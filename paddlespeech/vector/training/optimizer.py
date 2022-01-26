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
import paddle
from paddle import nn
from paddlespeech.s2t.training.scheduler import LRSchedulerFactory
from paddlespeech.s2t.training.optimizer import OptimizerFactory

def build_optimizers(model: nn.Layer,
                     config):
    if config["optim"].lower() == "adam":
        optim_type = config.optim
        optim_conf = config.optim_conf
        scheduler_type = config.scheduler
        scheduler_conf = config.scheduler_conf
        scheduler_args = {
            "learning_rate": optim_conf.lr,
            "verbose": False,
            "warmup_steps": scheduler_conf.warmup_steps,
            "gamma": scheduler_conf.lr_decay,
            "d_model": config.model_conf.attention_channels,
            }
        
        lr_scheduler = LRSchedulerFactory.from_args(scheduler_type,
                                                scheduler_args)
        def optimizer_args(
                config,
                parameters,
                lr_scheduler=None, ):
            train_config = config
            optim_type = train_config.optim
            optim_conf = train_config.optim_conf
            scheduler_type = train_config.scheduler
            scheduler_conf = train_config.scheduler_conf
            return {
                "grad_clip": train_config.global_grad_clip,
                "weight_decay": optim_conf.weight_decay,
                "learning_rate": lr_scheduler
                if lr_scheduler else optim_conf.lr,
                "parameters": parameters,
                "epsilon": 1e-9 if optim_type == 'noam' else None,
                "beta1": 0.9 if optim_type == 'noam' else None,
                "beat2": 0.98 if optim_type == 'noam' else None,
            }
        optimzer_args = optimizer_args(config, model.parameters(), lr_scheduler)
        optimizer = OptimizerFactory.from_args(optim_type, optimzer_args)
        # lr_scheduler = lr_scheduler
        return optimizer, lr_scheduler
    else:
        logger.error("no such optimizer model type: {}".format(config["opt_type"]))
        exit(-1)  
