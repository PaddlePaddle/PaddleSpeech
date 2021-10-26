# TODO(karita): add this to all the transform impl.
class TransformInterface:
    """Transform Interface"""

    def __call__(self, x):
        raise NotImplementedError("__call__ method is not implemented")

    @classmethod
    def add_arguments(cls, parser):
        return parser

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Identity(TransformInterface):
    """Identity Function"""

    def __call__(self, x):
        return x
