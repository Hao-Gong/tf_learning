#This is a program that classificate the mnist sample

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

        ##define the network objects
        self.conv1=nn.Sequential(
            nn.Conv2d(
                #in_channels means the input depth
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,  #confirm the size of conv the same size of input
            ), #-> (16,28,28)
            nn.ReLU(),
            #2*2 pooling size
            # -> (16,14,14)
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(
                #in_channels means the input depth
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1, #confirm the size of conv the same size of input
            ),#-> (32,14,14)
            nn.ReLU(),
            #2*2 pooling size
            nn.MaxPool2d(kernel_size=2),#-> (32,7,7)
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        #use the defined object to construct the true network
        x=self.conv1(x)#(batch,1,28,28)->(batch,16,14,14)
        x=self.conv2(x)#(batch,16,14,14)->(batch,32,7,7)
        #(32,7,7)->(batch,32*7*7)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output,x




EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False
#DOWNLOAD_MNIST=True

#load the train datasets and test datasets
train_data=torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(), #compress from 0-255 to (0,1)
    download=DOWNLOAD_MNIST
)

test_data=torchvision.datasets.MNIST(root='./mnist/', train=False)

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
#
# plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
# plt.title('%i'%train_data.train_labels[0])
# plt.show()

train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
# reshape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# only take the front 2000 datasets to check the accuracy
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

cnn=CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')