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

        ## RCNN
        self.roi_pool_conv5=nn.AvgPool2d(6)
        self.fc6 = nn.Linear(256*2*2, 4096)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(0.5)

        self.cls_score = nn.Linear(4096, 21)
        self.bbox_pred = nn.Linear(4096, 84)

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
        x=self.roi_pool_conv5(x)
        # 平铺x
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_score,bbox_pred



cnn=CNN()
out=cnn(Variable(torch.randn(1,3,224,224)))
print(out[0].size())
print(out[1].size())
