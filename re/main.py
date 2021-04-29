import torch
from geomloss import SamplesLoss
from torch.utils.data import random_split

result = random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
print(list(result[0]))