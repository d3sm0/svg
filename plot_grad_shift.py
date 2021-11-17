import torch
import torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(1,1,bias=False)
model.weight.data = model.weight.data *10.
x_0 = torch.randn((1,1))
optim = torch.optim.SGD(params=model.parameters(),lr=1e-2)
xs = [x_0]
x  = x_0
for _ in range(10):
    x = model(x).detach()
    xs.append(x)

gather_grads = []
for _ in range(10):
    x = x_0
    optim.zero_grad()
    for t in range(10):
        assert torch.allclose(x,xs[t])
        y =  model(x)
        x = y + xs[t+1] - y
        (y-xs[t]).pow(2).backward(retain_graph=True)
    grads = model.weight.grad
    gather_grads.append(grads)
    optim.step()
print(gather_grads)