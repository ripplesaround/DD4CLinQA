import torch
import sys
from AutoLabReport.AutoLabReport import AutoLabReport
import numpy as np


def qt_cal_density_matrix(x):
    # https://stackoverflow.com/questions/66894586/eig-cpu-not-implemented-for-long-in-torch-eigx
    outer_product = torch.outer(x, x)
    (evals, evecs) = torch.eig(outer_product, eigenvectors=True)
    density_matrix = torch.zeros(len(x),len(x),dtype=torch.float)
    for i,eigval in enumerate(evals):
        eigval = eigval[0]
        density_matrix += (eigval * torch.outer(evecs[:,i],evecs[:,i]))
    return density_matrix


if __name__== "__main__":

    # ALR = AutoLabReport()
    # ALR.get_text(text="你好")
    # ALR.build_pdf()

    a = torch.tensor([1,3,3,4.0])
    print(a)
    print(torch.outer(a,a))
    print(qt_cal_density_matrix(a))

    # X = torch.tensor([[25, 2, 9], [5, 25, -5], [3, 7, -1.0]])
    # e, v = torch.eig(X, eigenvectors=True)
    # print(e)
    # print(v)