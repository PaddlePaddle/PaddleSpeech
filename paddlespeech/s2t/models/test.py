import paddle
import paddle.nn as nn

class Model(nn.Layer):
    def __init__(self):
        super().__init__()        
        self.linear = nn.Linear(1024,1024)

    def forward(self, x):
        return self.linear(x)

model = Model()
x = paddle.uniform([100,1024], dtype='float32')
out = model(x)
loss = paddle.mean(out)
loss.backward()

clip = nn.ClipGradByGlobalNorm(clip_norm=1.0)
optim = paddle.optimizer.Adadelta(learning_rate=0.1, parameters=model.parameters(), grad_clip=clip)
optim.step()