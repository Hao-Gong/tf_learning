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

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #conv1
        self.conv1=nn.Conv2d(in_channels=3,
                        out_channels=96,
                        kernel_size=7,
                        stride=2,
                        padding=3)
        self.relu1=nn.ReLU()
        self.norm1=nn.BatchNorm2d(96)
        self.pooling1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #conv2
        self.conv2=nn.Conv2d(in_channels=96,
                        out_channels=256,
                        kernel_size=5,
                        stride=2,
                        padding=2)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(256)
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # conv3
        self.conv3=nn.Conv2d(in_channels=256,
                        out_channels=384,
                        kernel_size=3,
                        stride=1,
                        padding=1)
        self.relu3 = nn.ReLU()
        # conv4
        self.conv4=nn.Conv2d(in_channels=384,
                        out_channels=384,
                        kernel_size=3,
                        stride=1,
                        padding=1)
        self.relu4 = nn.ReLU()
        # conv5
        self.conv5=nn.Conv2d(in_channels=384,
                        out_channels=256,
                        kernel_size=3,
                        stride=1,
                        padding=1)
        self.relu5 = nn.ReLU()

        ## rpn
        self.rpn_conv1=nn.Conv2d(in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        stride=1,
                        padding=1)
        self.rpn_relu1 = nn.ReLU()
        self.rpn_cls_score=nn.Conv2d(in_channels=256,
                        out_channels=18,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.rpn_bbox_pred=nn.Conv2d(in_channels=256,
                        out_channels=36,
                        kernel_size=1,
                        stride=1,
                        padding=0)
    def forward(self,x):
        #conv1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pooling1(x) #torch.Size([1, 96, 56, 56])
        # conv2
        x = self.conv2(x)#torch.Size([1, 256, 28, 28])
        x = self.relu2(x)
        x = self.norm2(x)#torch.Size([1, 256, 28, 28])
        x = self.pooling2(x)#torch.Size([1, 256, 14, 14])
        # conv3
        x = self.conv3(x)
        x = self.relu3(x)
        # conv4
        x = self.conv4(x)
        x = self.relu4(x)
        # conv4
        x = self.conv5(x)
        x = self.relu5(x)

        ## rpn
        x = self.rpn_conv1(x)
        rpn_conv1 = self.rpn_relu1(x)
        rpn_cls_score = self.rpn_cls_score(rpn_conv1)
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv1)

        rpn_cls_score_reshape = rpn_cls_score.view(rpn_cls_score.size(0),
                                                   rpn_cls_score.size(1),2,-1)
        return rpn_cls_score,rpn_cls_score_reshape



cnn=CNN()
out=cnn(Variable(torch.randn(10,3,224,224)))
print(out[0].size())
print(out[1].size())
