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
import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import trainer, is_datasets_available
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# torch.distributed.init_process_group(backend="nccl")

class Trainer_CL(trainer.Trainer):
    def get_train_dataloader(self):
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        print("继承了类 %s   %s",100,self.train_dataset.__len__())
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        train_sampler = self._get_train_sampler()
        # 在这里添加一个id序列，让他经过相同的sample操作，得到变换后的id序列
        # self.train_dataset.__len__()
        result = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            # shuffle=True,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            # generator=g,
        )
        return result
