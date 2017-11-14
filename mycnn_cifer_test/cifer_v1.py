from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = pickle.load(f, encoding='bytes')
        batch_label = datadict[b'batch_label']
        data = datadict[b'data']
        filename=datadict[b'filenames']
        labels=datadict[b'labels']
        return batch_label, data,filename,labels

NUMBER=0


#batch_label, data,filename,labels = load_CIFAR_batch("/root/pytorch_learning/tf_learning/cifar-10-python/cifar-10-batches-py/test_batch")
batch_label, data,filename,labels = load_CIFAR_batch("/home/gong/tf_learning/cifar-10-python/cifar-10-batches-py/test_batch")

image1=data[NUMBER].reshape(3,32,32)

i0 = Image.fromarray(image1[0])
i1 = Image.fromarray(image1[1])
i2 = Image.fromarray(image1[2])
img = Image.merge("RGB", (i0, i1, i2))

print("image_data:",img)
print("image_label:",labels[NUMBER])
print("name:",filename[NUMBER])
#img.show()

plt.imshow(img)
plt.title('%s'%filename[NUMBER])
plt.show()