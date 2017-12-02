import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F


# Vector=np.array([10,-1,6,1,-2],dtype=float)
Vector=np.array([1,1,1,1,1],dtype=float)
Vector_pt=autograd.Variable(torch.from_numpy(Vector))
print("Original Tensor:",Vector_pt)
Probs = F.softmax(Vector_pt)

print("Sotfmax posibility:",Probs)

#log(1/p(i))=-log(p(i)),pytorch有直接算出Softmax再log的
log_entropy=-F.log_softmax(Vector_pt)
# print(log_entropy)
# 每个元素相乘
Information_entropy=Probs*log_entropy
print("Information Entropy:",sum(Information_entropy))