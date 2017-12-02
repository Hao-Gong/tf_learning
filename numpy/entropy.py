import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

#设置一个概率分布,其实里面数值随意的，经过softmax都会归一化
Vector=autograd.Variable(torch.from_numpy(np.array([0.3,0.1,0.1,0.1,0.4],dtype=float)))

#我们期望的概率分布是，第二个元素是1,分布向两为[0,1,0,0,0]
Target=autograd.Variable(torch.from_numpy(np.array([0,1,0,0,0],dtype=float)))
# print(Target)
#
# loss_func = nn.CrossEntropyLoss()
#
# loss=loss_func(Vector, Target)
#
# print("Cross_Entropy:",loss)