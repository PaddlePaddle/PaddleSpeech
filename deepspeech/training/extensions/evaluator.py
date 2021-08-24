from typing import Dict

import paddle
from paddle.io import DataLoader
from paddle.nn import Layer

import extension
from ..reporter import DictSummary
from ..reporter import report
from ..reporter import scope


class StandardEvaluator(extension.Extension):

    trigger = (1, 'epoch')
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, model: Layer, dataloader: DataLoader):
        # it is designed to hold multiple models
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        self.model = model

        # dataloaders
        self.dataloader = dataloader

    def evaluate_core(self, batch):
        # compute
        self.model(batch)  # you may report here

    def evaluate(self):
        # switch to eval mode
        for model in self.models.values():
            model.eval()

        # to average evaluation metrics
        summary = DictSummary()
        for batch in self.dataloader:
            observation = {}
            with scope(observation):
                # main evaluation computation here.
                with paddle.no_grad():
                    self.evaluate_core(batch)
            summary.add(observation)
        summary = summary.compute_mean()
        return summary

    def __call__(self, trainer=None):
        # evaluate and report the averaged metric to current observation
        # if it is used to extend a trainer, the metrics is reported to
        # to observation of the trainer
        # or otherwise, you can use your own observation
        summary = self.evaluate()
        for k, v in summary.items():
            report(k, v)