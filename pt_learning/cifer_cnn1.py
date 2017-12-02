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


def ConverToTensor(data):
    data_list=[]
    for i in range(10000):
        img_tensor = torch.FloatTensor(data[0][i] / 255.)
        img_tuple = (img_tensor, data[1][i])
        data_list.append(img_tuple)
    return data_list


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
        self.out=nn.Linear(96*8*8,10)

    def forward(self,x):
        #use the defined object to construct the true network
        x=self.conv1(x)#(batch,3,32,32)->(batch,48,16,16)
        x=self.conv2(x)#(batch,48,16,16)->(batch,96,8,8)
        #(96,8,8)->(batch,96*8*8)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output


########################################################################################################################
###define the global parameter
########################################################################################################################
EPOCH=1
TRAIN_BATCH_SIZE=100
TEST_BATCH_SIZE=500
LR=0.01
########################################################################################################################

# batch_label, data, filename, labels = load_CIFAR_batch("/root/pytorch_learning/tf_learning/cifar-10-python/cifar-10-batches-py/test_batch")
##load the train data set
#data type
data = load_CIFAR_batch("/home/gong/tf_learning/cifar-10-python/cifar-10-batches-py/data_batch_1")
train_data=ConverToTensor(data)
# print(np.shape(train_data))
train_loader=Data.DataLoader(dataset=train_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)

# ##load the test data set
data_t = load_CIFAR_batch("/home/gong/tf_learning/cifar-10-python/cifar-10-batches-py/test_batch")
test_data=ConverToTensor(data_t)
test_loader=Data.DataLoader(dataset=test_data,batch_size=TEST_BATCH_SIZE,shuffle=True)

cnn=CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    # gives batch data, normalize x when iterate train_loader
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y
        # print(b_x.size())
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        #print("step:",step)
        if step % 50 == 0:
            for i_test, (x_test, y_test) in enumerate(test_loader):
                x_t = Variable(x_test)
                y_t = Variable(y_test)
                print(x_t.size())
                test_output = cnn(x_t)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == y_test) / float(y_test.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
                break