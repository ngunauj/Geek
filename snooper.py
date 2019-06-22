import torch
import torchsnooper

# @torchsnooper.snoop()
# def myfunc(mask, x):
#     y = torch.zeros(6, device='cuda')
#     y.masked_scatter_(mask, x) #tihuan
#     return y
#
# mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda', dtype=torch.uint8)
# source = torch.tensor([1.0, 2.0, 3.0], device='cuda')
# y = myfunc(mask, source)

model = torch.nn.Linear(2, 1)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([3.0, 5.0, 4.0, 6.0])
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
with torchsnooper.snoop():
    for _ in range(10):
        optimizer.zero_grad()
        pred = model(x).squeeze()
        squared_diff = (y - pred) ** 2
        loss = squared_diff.mean()
        print(loss.item())
        loss.backward()
        optimizer.step()