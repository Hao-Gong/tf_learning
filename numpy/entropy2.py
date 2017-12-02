import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

q=autograd.Variable(torch.from_numpy(np.array([1,10,1,1,1],dtype=float)))
#设置一个真实概率分布,其实里面数值随意的，经过softmax都会归一化
print("q:",F.softmax(q))
#我们期望假设的概率分布是，第二个元素是1
p=autograd.Variable(torch.from_numpy(np.array([0,1,0,0,0],dtype=float)))
print("p:",p)
#计算log(1/q(i))
log_entropy=-F.log_softmax(q)

#计算p(i)*log(1/q(i))
cross_entropy=p*log_entropy
print("Cross Entropy:",sum(cross_entropy))

# import numpy as np
# import torch
# import torch.autograd as autograd
# import torch.nn.functional as F
# import torch.nn as nn
#
# q_Val=autograd.Variable(torch.from_numpy(np.array([1,10,1,1,1],dtype=float)))
# #设置一个真实概率分布
# q=F.softmax(q_Val)
# #我们期望假设的概率分布是，第二个元素是1
# p=autograd.Variable(torch.from_numpy(np.array([0,1,0,0,0],dtype=float)))
#
# #计算log(1/q(i))
# q_log=-F.log_softmax(q_Val)
# #计算log(p(i))
# p_log=p.log()
#
# print(p.log())
# #计算p(i)*log(p(i)/q(i))
# relative_entropy=p*p_log*q_log
# print("Relative Entropy:",sum(relative_entropy))
#
# cross_entropy=p*(-F.log_softmax(q_Val))
# Information_entropy=F.softmax(q_Val)*(-F.log_softmax(q_Val))
# print(sum(cross_entropy)-sum(Information_entropy))