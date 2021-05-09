import torch
import sys
from AutoLabReport.AutoLabReport import AutoLabReport
import numpy as np

if __name__== "__main__":

    # ALR = AutoLabReport()
    # ALR.get_text(text="你好")
    # ALR.build_pdf()

    print('Python: {}'.format(sys.version))
    print(torch.__version__)
    re1 = [92.32,	52.1,88.64,65.7,56.34 ,	81.61,88.85]
    re2 = [91.86,56.53,89.06 ,63.18,35.21 ,81.62 ,	87.35 ]
    re3 = [92.32,59.6,88.44,65.7,56.3,	82.35,87.23 ]
    print(np.mean(re1))
    print(np.mean(re2))
    print(np.mean(re3))