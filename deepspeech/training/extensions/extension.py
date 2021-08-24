from typing import Callable

PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100


class Extension():
    """Extension to customize the behavior of Trainer."""
    trigger = (1, 'iteration')
    priority = PRIORITY_READER
    name = None

    @property
    def default_name(self):
        """Default name of the extension, class name by default."""
        return type(self).__name__

    def __call__(self, trainer):
        """Main action of the extention. After each update, it is executed
        when the trigger fires."""
        raise NotImplementedError(
            'Extension implementation must override __call__.')

    def initialize(self, trainer):
        """Action that is executed once to get the corect trainer state.
        It is called before training normally, but if the trainer restores
        states with an Snapshot extension, this method should also be called.
        """
        pass

    def on_error(self, trainer, exc, tb):
        """Handles the error raised during training before finalization.
        """
        pass

    def finalize(self, trainer):
        """Action that is executed when training is done.
        For example, visualizers would need to be closed.
        """
        pass