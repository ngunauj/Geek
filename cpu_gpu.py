import time
import torch
print(torch.__version__)
print(torch.cuda.is_available)

a = torch.randn(1000, 10000)
b = torch.randn(10000, 1000)

t0 = time.time()
c =  torch.matmul(a, b)
t1 = time.time()
print('cpu time:{} c={}'.format(t1 - t0, c))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)
t0 = time.time()
c =  torch.matmul(a, b)
t1 = time.time()
print('gpu time:{} c={}'.format(t1 - t0, c))