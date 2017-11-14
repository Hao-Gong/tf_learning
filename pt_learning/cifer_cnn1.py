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
        batch_label = datadict[b'batch_label']
        data = datadict[b'data'].reshape(10000, 3, 32, 32)
        filename=datadict[b'filenames']
        labels=datadict[b'labels']
        return batch_label, data,filename,labels

def ConvetToImg(data):
    i0 = Image.fromarray(data[0])
    i1 = Image.fromarray(data[1])
    i2 = Image.fromarray(data[2])
    return Image.merge("RGB", (i0, i1, i2))

NUMBER=0


batch_label, data,filename,labels = load_CIFAR_batch("/root/pytorch_learning/tf_learning/cifar-10-python/cifar-10-batches-py/test_batch")
# batch_label, data,filename,labels = load_CIFAR_batch("/home/gong/tf_learning/cifar-10-python/cifar-10-batches-py/test_batch")

# image1=data[NUMBER]
# image2=data[NUMBER+1]
# print(image1)
#
# img1 = ConvetToImg(image1)
# img2 = ConvetToImg(image2)
# # print("image_data:",image1)
# # print("image_label:",labels[NUMBER])
# # print("name:",filename[NUMBER])
# plt.subplot(131)
# plt.imshow(img1)
# plt.subplot(132)
# plt.imshow(img2)
# plt.show()




