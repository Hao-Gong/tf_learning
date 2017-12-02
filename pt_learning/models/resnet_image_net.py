import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from resnet import resnet
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

# resnet.test()


IMAGENET_TEST_PATH="/home/gong/tiny-imagenet-200/test/images/"

image_test=Image.open(IMAGENET_TEST_PATH+"test_0.JPEG")
# plt.imshow(image_test)
# plt.show()
image_test_trans=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ])
image_test_tensor=torch.unsqueeze(image_test_trans(image_test),0)
image_test_Val=Variable(image_test_tensor)
# print(image_test_tensor[0][0])
print(image_test_Val.size())
net = resnet.resnet18(pretrained=True)
print(net)
result=net(Variable(torch.randn(1,3,224,224)))
print(result)