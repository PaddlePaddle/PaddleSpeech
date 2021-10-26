import numpy


class ChannelSelector():
    """Select 1ch from multi-channel signal"""

    def __init__(self, train_channel="random", eval_channel=0, axis=1):
        self.train_channel = train_channel
        self.eval_channel = eval_channel
        self.axis = axis

    def __repr__(self):
        return (
            "{name}(train_channel={train_channel}, "
            "eval_channel={eval_channel}, axis={axis})".format(
                name=self.__class__.__name__,
                train_channel=self.train_channel,
                eval_channel=self.eval_channel,
                axis=self.axis,
            )
        )

    def __call__(self, x, train=True):
        # Assuming x: [Time, Channel] by default

        if x.ndim <= self.axis:
            # If the dimension is insufficient, then unsqueeze
            # (e.g [Time] -> [Time, 1])
            ind = tuple(
                slice(None) if i < x.ndim else None for i in range(self.axis + 1)
            )
            x = x[ind]

        if train:
            channel = self.train_channel
        else:
            channel = self.eval_channel

        if channel == "random":
            ch = numpy.random.randint(0, x.shape[self.axis])
        else:
            ch = channel

        ind = tuple(slice(None) if i != self.axis else ch for i in range(x.ndim))
        return x[ind]
