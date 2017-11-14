import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

BATCH_SIZE=8

#x 1,2,3...,9,10
x=torch.linspace(1,10,10)
#y 10,9,8....1
y=torch.linspace(10,1,10)
torch_dataset=Data.TensorDataset(data_tensor=x,target_tensor=y)


#shuffle=True means the mini batch are random
#num_worker means the threads to use
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

for epoch in range(3):
    #enumerate
    for step,(batch_x,batch_y) in enumerate(loader):
        #training..
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
