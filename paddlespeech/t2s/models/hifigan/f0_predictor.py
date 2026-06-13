import paddle
class ConvRNNF0Predictor(paddle.nn.Layer):

    def __init__(self, num_class: int=1, in_channels: int=80, cond_channels:
        int=512):
        super().__init__()
        self.num_class = num_class
        self.condnet = paddle.nn.Sequential(
            paddle.nn.Conv1D(in_channels, cond_channels, kernel_size=3, padding=1), 
            paddle.nn.ELU(), 
            paddle.nn.Conv1D(cond_channels, cond_channels,kernel_size=3, padding=1), 
            paddle.nn.ELU(), 
            paddle.nn.Conv1D(cond_channels, cond_channels,kernel_size=3, padding=1), 
            paddle.nn.ELU(), 
            paddle.nn.Conv1D(cond_channels, cond_channels,kernel_size=3, padding=1), 
            paddle.nn.ELU(), 
            paddle.nn.Conv1D(cond_channels, cond_channels,kernel_size=3, padding=1), 
            paddle.nn.ELU()
        )
        self.classifier = paddle.nn.Linear(in_features=cond_channels,
            out_features=self.num_class)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        for idx,layer in enumerate(self.condnet):
            x = layer(x)
        x = paddle.transpose(x, perm=[0, 2, 1])
        return paddle.abs(x=self.classifier(x).squeeze(-1))
