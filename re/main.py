import torch
from geomloss import SamplesLoss

X_i = torch.randn([10,10])
Y_j = torch.randn([10,10])

blur = .05
OT_solver = SamplesLoss("sinkhorn", p=2, blur=blur,
                        scaling=.9, debias=False, potentials=True)
F_i, G_j = OT_solver(X_i, Y_j)

print(F_i)
print(G_j)
