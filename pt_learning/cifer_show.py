#使用pickle工具加载CIFAR-10的batch包
import pickle
#可以使用PIL来显示图片，也可以使用plt来显示图片
# 本人推荐plt，显示的时候可以自动调节大小，CIFAR-10图片相当小
from PIL import Image
import matplotlib.pyplot as plt

#加载单个CIFAR包
def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        #加载出来的数据是一个10000大小的字典，有'batch_label'，'data'，'filenames'，'labels'
        datadict = pickle.load(f, encoding='bytes')
        #'batch_label'是0-9的数字
        batch_label = datadict[b'batch_label']
        #'data'是图片的数据，numpy.ndarray格式，需要从(10000,3,32×32)->(10000, 3, 32, 32)
        data = datadict[b'data'].reshape(10000, 3, 32, 32)
        #图片名字，用于保存图像使用
        filename=datadict[b'filenames']
        #图片的序号，第几张图片
        labels=datadict[b'labels']
        return data, labels, batch_label, filename

#需要将图片读出到RGB通道，混合形成RGB图片
def ConvetToImg(data):
    i0 = Image.fromarray(data[0])
    i1 = Image.fromarray(data[1])
    i2 = Image.fromarray(data[2])
    return Image.merge("RGB", (i0, i1, i2))

#第几张图片
SAMPLE_NUM=0
#你的cifar文件夹
CIFAR_PATH="/home/gong/tf_learning/cifar-10-python/cifar-10-batches-py/data_batch_1"

data = load_CIFAR_batch(CIFAR_PATH)

img=ConvetToImg(data[0][SAMPLE_NUM])

plt.imshow(img)

plt.text(2, -3, "image name:  %s  "%data[3][SAMPLE_NUM], fontdict={'size': 8, 'color':  'red'})
plt.text(2, -1, "image label:  %s  "%data[1][SAMPLE_NUM], fontdict={'size': 8, 'color':  'red'})
plt.show()

#也可以使用这个显示，不过图片相当小，是原始像素尺寸
# img.show()