# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import matplotlib.image as plimg
from PIL import Image

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y

def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        lines = [x for x in f.readlines()]
        print(lines)


if __name__ == "__main__":
    load_CIFAR_Labels("../cifar-10-python/cifar-10-batches-py/batches.meta")
    imgX, imgY = load_CIFAR_batch("../cifar-10-python/cifar-10-batches-py/data_batch_1")
    for i in xrange(imgX.shape[0]):
        imgs = imgX[i - 1]
        if i < 100:
            img0 = imgs[0]
            img1 = imgs[1]
            img2 = imgs[2]
            i0 = Image.fromarray(img0)
            i1 = Image.fromarray(img1)
            i2 = Image.fromarray(img2)
            img = Image.merge("RGB",(i0,i1,i2))
            name = "img" + str(i)
            img.save("/data/images/"+name,"png")#文件夹下是RGB融合后的图像
            for j in xrange(imgs.shape[0]):
                img = imgs[j - 1]
                name = "img" + str(i) + str(j) + ".png"
                plimg.imsave("/data/image/" + name, img)#文件夹下是RGB分离的图像

