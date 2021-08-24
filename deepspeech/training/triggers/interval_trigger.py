
class IntervalTrigger():
    """A Predicate to do something every N cycle."""

    def __init__(self, period: int, unit: str):
        if unit not in ("iteration", "epoch"):
            raise ValueError("unit should be 'iteration' or 'epoch'")
        if period <= 0:
            raise ValueError("period should be a positive integer.")
        self.period = period
        self.unit = unit
        self.last_index = None

    def __call__(self, trainer):
        if self.last_index is None:
            last_index = getattr(trainer.updater.state, self.unit)
            self.last_index = last_index

        last_index = self.last_index
        index = getattr(trainer.updater.state, self.unit)
        fire = index // self.period != last_index // self.period

        self.last_index = index
        return fire