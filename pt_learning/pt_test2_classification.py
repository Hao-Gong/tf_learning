import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


#class Hidden_layer definieren
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        #only two layers
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

#n_data is a matrix 100*2
n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
print(x0)
#label of the data set 0 is 0
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
#label of the data set 1 is 1
y1=torch.ones(100)

#FloatTensor = 32-bit float, torch.cat means comblie two matrix into one
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
#LongTensor = 64-bit integer
y=torch.cat((y0,y1),0).type(torch.LongTensor)

x,y=Variable(x),Variable(y)


#plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
#plt.show()

net=Net(n_feature=2, n_hidden=10, n_output=2)
print(net) # net architecture


optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()   # something about plotting

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 20 == 0:
        # plot and show learning process
        plt.cla()
        #print(out)
        prediction = torch.max(F.softmax(out), 1)[1]
        #print(torch.max(F.softmax(out), 1))[0]
        #print(prediction)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
