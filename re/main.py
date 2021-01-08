import torch
import torch.nn as nn
import torch.nn.functional as F
x = torch.rand(10)
y = torch.rand(10)
x = F.log_softmax(x)
y = F.softmax(y)

criterion = nn.KLDivLoss()
klloss = criterion(x, y)
print("here")
print(klloss)
