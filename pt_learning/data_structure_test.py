from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
import torchvision

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = pickle.load(f, encoding='bytes')
        # batch_label = datadict[b'batch_label']
        data = datadict[b'data'].reshape(10000, 3, 32, 32)
        # data = datadict[b'data']
        # filename=datadict[b'filenames']
        labels=datadict[b'labels']
        return data, labels

def ConvetToImg(data):
    i0 = Image.fromarray(data[0])
    i1 = Image.fromarray(data[1])
    i2 = Image.fromarray(data[2])
    return Image.merge("RGB", (i0, i1, i2))




class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        ##define the network objects
        #input size (3,32,32)
        self.conv1=nn.Sequential(
            nn.Conv2d(
                #in_channels means the input depth
                in_channels=3,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1,  #confirm the size of conv the same size of input
            ), #-> (48,32,32)
            nn.ReLU(),
            #2*2 pooling size
            # -> (48,16,16)
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(
                #in_channels means the input depth
                in_channels=48,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1, #confirm the size of conv the same size of input
            ),#-> (96,16,16)
            nn.ReLU(),
            #2*2 pooling size
            # -> (96,8,8)
            nn.MaxPool2d(kernel_size=2),
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        #use the defined object to construct the true network
        x=self.conv1(x)#(batch,3,32,32)->(batch,48,16,16)
        x=self.conv2(x)#(batch,48,16,16)->(batch,96,8,8)
        #(32,7,7)->(batch,32*7*7)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output,x


########################################################################################################################
###define the global parameter
########################################################################################################################
EPOCH=1
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1000
LR=0.001
########################################################################################################################

data = load_CIFAR_batch("/home/gong/tf_learning/cifar-10-python/cifar-10-batches-py/data_batch_1")

img1=torch.FloatTensor(data[0][0]/255.)
img2=torch.FloatTensor(data[0][1]/255.)
img3=torch.FloatTensor(data[0][2]/255.)
l1=(img1,data[1][0])
l2=(img2,data[1][1])
l3=(img3,data[1][2])
L=[]
L.append(l1)
L.append(l2)
L.append(l3)

# print(L)

# t1=([1,1],1)
# t2=([2,2],2)
# t3=([3,3],3)
# L.append(t1)
# L.append(t2)
# L.append(t3)
# # print(type(L[0][0]))
# data=torch.FloatTensor(L)
train_loader=Data.DataLoader(dataset=L,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
#
for epoch in range(EPOCH):
    # gives batch data, normalize x when iterate train_loader
    for step, (x ,y) in enumerate(train_loader):
        b_x = Variable(x)   # batch x
        b_y = Variable(y)
        #print()
        print(b_y)
