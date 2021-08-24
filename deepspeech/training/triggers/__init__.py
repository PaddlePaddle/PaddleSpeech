from .interval_trigger import IntervalTrigger

def never_fail_trigger(trainer):
    return False

def get_trigger(trigger):
    if trigger is None:
        return never_fail_trigger
    if callable(trigger):
        return trigger
    else:
        trigger = IntervalTrigger(*trigger)
        return trigger