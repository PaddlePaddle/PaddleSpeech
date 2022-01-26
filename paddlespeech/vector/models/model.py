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
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.models.ecapa_tdnn import AdditiveAngularMargin
from paddlespeech.vector.models.ecapa_tdnn import LogSoftmaxWrapper
from paddlespeech.vector.models.ecapa_tdnn import CosClassifier

logger = Log(__name__).getlog()

def build_sid_models(config):
    if config["speaker_model"].lower() == "ecapa-xvector":
        model = EcapaTdnn.init_from_config(config)
    else:
        logger.error("no such speaker model type: {}".format(config["speaker_model"]))
        exit(-1)
    
    return model

def build_sid_loss(config):
    if config["loss_type"].lower() == "logsoftmaxwrapper":
        if config["loss_conf"]["loss_fn"].lower() == "additiveangularmargin":
            margin = config["loss_conf"]["margin"]
            scale = config["loss_conf"]["scale"]
            loss_fn = AdditiveAngularMargin(margin=margin, scale=scale)
        else:
            logger.error("no such loss pre-function : {}".format(config["loss_conf"]["loss_fn"]))
            exit(-1)                 
        loss_fn = LogSoftmaxWrapper(loss_fn)
    else:
        logger.error("no such classifier model type: {}".format(config["classifier_type"]))
        exit(-1)  

    return loss_fn


def build_sid_classifier(config):
    if config["classifier_type"].lower() == "cosclassifier":
        classifier = CosClassifier.init_from_config(config)
    else:
        logger.error("no such classifier model type: {}".format(config["classifier_type"]))
        exit(-1)  

    return classifier
