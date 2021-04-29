# # coding=utf-8
#
# '''
# Author: ripples
# Email: ripplesaround@sina.com
#
# date: 2021/4/26 00:56
# desc:
# '''

# import os
# # notice 制定GPU
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# torch.distributed.init_process_group(backend="nccl")

input_size = 5
output_size = 2
batch_size = 2
data_size = 16

local_rank = 0
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


class RandomDataset(Dataset):
    def __init__(self, size, length, local_rank):
        self.len = length
        self.data = torch.stack([torch.ones(5), torch.ones(5) * 2,
                                 torch.ones(5) * 3, torch.ones(5) * 4,
                                 torch.ones(5) * 5, torch.ones(5) * 6,
                                 torch.ones(5) * 7, torch.ones(5) * 8,
                                 torch.ones(5) * 9, torch.ones(5) * 10,
                                 torch.ones(5) * 11, torch.ones(5) * 12,
                                 torch.ones(5) * 13, torch.ones(5) * 14,
                                 torch.ones(5) * 15, torch.ones(5) * 16]).to('cuda')

        self.local_rank = local_rank

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


dataset = RandomDataset(input_size, data_size, local_rank)
sampler = SequentialSampler(dataset)
rand_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         sampler=sampler)

e = 0
while e < 2:
    t = 0
    # sampler.set_epoch(e)
    for data in rand_loader:
        print(data)
    e += 1