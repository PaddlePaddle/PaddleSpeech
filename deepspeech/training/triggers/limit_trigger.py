
class LimitTrigger():
    """A Predicate to decide whether to stop."""

    def __init__(self, limit: int, unit: str):
        if unit not in ("iteration", "epoch"):
            raise ValueError("unit should be 'iteration' or 'epoch'")
        if limit <= 0:
            raise ValueError("limit should be a positive integer.")
        self.limit = limit
        self.unit = unit

    def __call__(self, trainer):
        state = trainer.updater.state
        index = getattr(state, self.unit)
        fire = index >= self.limit
        return fire