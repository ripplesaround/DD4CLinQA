import torch
import torch.nn as nn
import torch.nn.functional as F

subset_quantity = 3
subset_id = [1,5,6,8,0]
print(subset_id[0 : int((len(subset_id)/subset_quantity))])
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                subset_id[0 : int((len(subset_id)/subset_quantity))]
            )
print(len(train_sampler))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                subset_id
            )
print(len(train_sampler))