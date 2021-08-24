from deepspeech.training.extensions import extension
from deepspeech.training.updaters.trainer import Trainer


class VisualDL(extension.Extension):
    """A wrapper of visualdl log writer. It assumes that the metrics to be visualized
    are all scalars which are recorded into the `.observation` dictionary of the
    trainer object. The dictionary is created for each step, thus the visualdl log
    writer uses the iteration from the updater's `iteration` as the global step to
    add records.
    """
    trigger = (1, 'iteration')
    default_name = 'visualdl'
    priority = extension.PRIORITY_READER

    def __init__(self, writer):
        self.writer = writer

    def __call__(self, trainer: Trainer):
        for k, v in trainer.observation.items():
            self.writer.add_scalar(k, v, step=trainer.updater.state.iteration)

    def finalize(self, trainer):
        self.writer.close()