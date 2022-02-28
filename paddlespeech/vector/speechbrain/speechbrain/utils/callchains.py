"""Chaining together callables, if some require relative lengths"""
import inspect


def lengths_arg_exists(func):
    """Returns True if func takes ``lengths`` keyword argument.

    Arguments
    ---------
    func : callable
        The function, method, or other callable to search for the lengths arg.
    """
    spec = inspect.getfullargspec(func)
    return "lengths" in spec.args + spec.kwonlyargs


class LengthsCapableChain:
    """Chain together callables. Can handle relative lengths.

    This is a more light-weight version of
    speechbrain.nnet.containers.LengthsCapableSequential

    Arguments
    ---------
    *funcs : list, optional
        Any number of functions or other callables, given in order of
        execution.

    Returns
    -------
    Any
        The input as processed by each function. If no functions were given,
        simply returns the input.
    """

    def __init__(self, *funcs):
        self.funcs = []
        self.takes_lengths = []
        for func in funcs:
            self.append(func)

    def __call__(self, x, lengths=None):
        """Run the chain of callables on the given input

        Arguments
        ---------
        x : Any
            The main input
        lengths : Any
            The lengths argument which will be conditionally passed to
            any functions in the chain that take a 'lengths' argument.
            In SpeechBrain the convention is to use relative lengths.

        Note
        ----
        By convention, if a callable in the chain returns multiple outputs
        (returns a tuple), only the first output is passed to the next
        callable in the chain.
        """
        if not self.funcs:
            return x
        for func, give_lengths in zip(self.funcs, self.takes_lengths):
            if give_lengths:
                x = func(x, lengths)
            else:
                x = func(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def append(self, func):
        """Add a function to the chain"""
        self.funcs.append(func)
        self.takes_lengths.append(lengths_arg_exists(func))

    def __str__(self):
        clsname = self.__class__.__name__
        if self.funcs:
            return f"{clsname}:\n" + "\n".join(str(f) for f in self.funcs)
        else:
            return f"Empty {clsname}"
