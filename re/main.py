# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random

import torch


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    train_dataset = [1, 5, 78, 9, 68]
    n_train = len(train_dataset)
    split = n_train // 3
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    # b = torch.utils.data.RandomSampler(data_source = a, replacement=True,num_samples = int(len(a)/2))
    for x in train_sampler:
        print(train_dataset[x])
    print("len ",train_sampler.__len__())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
