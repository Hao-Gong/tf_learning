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
print(n_data)
x0=torch.normal(2*n_data,1)
#label of the data set 0 is 0
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
#label of the data set 1 is 1
y1=torch.ones(100)

#FloatTensor = 32-bit
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
#LongTensor = 64-bit
y=torch.cat((y0,y1),).type(torch.LongTensor)