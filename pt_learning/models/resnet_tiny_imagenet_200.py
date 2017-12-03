import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import os
import numpy as np
import random



#定义一个3×3的卷积层，后面多次调用，方便使用，尺寸不变，plane是深度的意思
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#BasicBlock是基本的Resnet结构，搭建18层和34层的Resnet，输入深度和输出深度不变
class BasicBlock(nn.Module):
    #输出深度/输入深度
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #保存x，后面用于融合
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #当输入深度不等于输出深度，定义downsample为内核为1的卷积层加上正则层，扩大residual使之匹配x的大小
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 输出深度/输入深度,每次调用这个，深度增长4倍
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #深度增长4倍，源码self.expansion直接写了4，改了这个，就可以自己调整了
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#定义整个网络
class ResNet(nn.Module):
    #block就是你要调用的基本单元BasicBlock/Bottleneck
    #layer is [,,,]，其中的元素就是4次构建基本模块的时候分别调用几次
    def __init__(self, block, layers, num_classes=200):
        self.inplanes = 64
        super(ResNet, self).__init__()
        #[,3,image_size,image_size]->[,64,image_size/2,image_size/2]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #[,64,image_size/2,image_size/2]->[,64,image_size/4,image_size/4]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ## _make_layer就是利用基本单元BasicBlock/Bottleneck构建网络
        self.layer1 = self._make_layer(block, 64, layers[0])
        # [,64,image_size/4,image_size/4]->[,128,image_size/8,image_size/8]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # [,128,image_size/8,image_size/8]->[,256,image_size/16,image_size/16]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # [,256,image_size/16,image_size/16]->[,512,image_size/32,image_size/32]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #平均池化层
        # [,512,image_size/32,image_size/32]->[,512,image_size/224,image_size/224]
        self.avgpool = nn.AvgPool2d(2, stride=1)
        # [, 512, image_size / 224, image_size / 224]->[,1000]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #遍历神经网络，初始化参数
        for m in self.modules():
            #如果遇到nn.Conv2d，则正则化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            #如果遇到nn.BatchNorm2d，将其weight填充1,bias清零
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #如果卷积层需要扩大block.expansion倍或者stride=2，则定义downsample为内核为1的卷积层加上正则层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        #nn.Sequential可以用layers list构建，简单方便
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #
        x = self.avgpool(x)
        #平铺x
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


#读取train_data,输出格式[path,(label,x_s,y_s,x_e,y_e)]
def read_train_data():
    image_list = []

    with open(TINY_PATH_ROOT + 'wnids.txt', 'r') as f:
        data = f.read()
    file_name = data.split()

    for i in range(200):
        with open(TINY_PATH_TRAIN + file_name[i] + '/' + file_name[i] + '_boxes.txt', 'r') as f:
            data = f.read()
        image_name_data = data.split()
        for j in range(500):
            image_path = TINY_PATH_TRAIN + file_name[i] + '/images/' + image_name_data[j * 5]
            x_s = int(image_name_data[j * 5 + 1])
            y_s = int(image_name_data[j * 5 + 2])
            x_e = int(image_name_data[j * 5 + 3])
            y_e = int(image_name_data[j * 5 + 4])
            tuple = np.array([x_s, y_s, x_e, y_e])
            image_list.append([image_path, i, tuple])
    random.shuffle(image_list)
    return image_list

def check_grey(im_t):
    if(im_t.size(0)==3):
        return im_t
    else:
        return torch.cat((im_t,im_t,im_t),0)


def train_batch_load(batch_size=50):
    image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    for cursor in range(0, len(image_train), batch_size):
        image_batch = torch.unsqueeze(check_grey(image_trans(Image.open(image_train[cursor][0]))), 0)
        label_batch = torch.torch.LongTensor([image_train[cursor][1]])
        box_batch = torch.unsqueeze(torch.from_numpy(image_train[cursor][2]), 0)
        batch=[]
        for i in range(1,batch_size):
            image = torch.unsqueeze(check_grey(image_trans(Image.open(image_train[cursor+i][0]))), 0)
            label = torch.torch.LongTensor([image_train[cursor+i][1]])
            box = torch.unsqueeze(torch.from_numpy(image_train[cursor+i][2]), 0)
            image_batch = torch.cat((image_batch, image), 0)
            label_batch = torch.cat((label_batch, label), 0)
            box_batch = torch.cat((box_batch, box), 0)
        batch.append(image_batch)
        batch.append(label_batch)
        batch.append(box_batch)
        print(cursor)
        yield batch

TINY_PATH_ROOT='/home/gong/tiny-imagenet-200/'
TINY_PATH_TRAIN='/home/gong/tiny-imagenet-200/train/'

BATCH_SIZE=200
image_train=read_train_data()

resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
# print(resnet18(Variable(torch.randn(50,3,64,64))).size())

optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()

# print(cnn(Variable(torch.randn(1,3,64,64))))
for epoch in range(1):
    for batch in train_batch_load(batch_size=BATCH_SIZE):
        # print(batch[0].size())
        b_x = Variable(batch[0])   # batch x
        b_y = Variable(batch[1])   # batch y
        # print(b_y)
        # print(b_x.size())
        output = resnet18(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        # print(type(batch[1]))
        pred_y = torch.max(output, 1)[1].data.squeeze()
        accuracy = sum(pred_y == batch[1]) / float(BATCH_SIZE)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
