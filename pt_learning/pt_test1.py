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


#dim=0 it is a line vector, dim=1 makes a raw vector
x= torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
#y=x^2+noise
y=x.pow(2)+0.2*torch.rand(x.size())

x,y=torch.autograd.Variable(x), Variable(y)

#start draw the plt
plt.scatter(x.data.numpy(),y.data.numpy())
plt.ion()

#object net, hidden_layersize=10
net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net) # net architecture

#optimizer define, SGD method,learning rate is 0.5
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# this is for regression mean squared loss
loss_func = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 50 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()