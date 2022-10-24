import sys
import torch.nn as nn
import torch
import numpy as np

dn = 50.
h=10
nepochs=100

with open("2019.csv","r") as f: ls=f.readlines()
trainx = torch.Tensor([float(l.split(',')[1])/dn for l in ls[:-1]]).view(1,-1,1)
trainy = torch.Tensor([float(l.split(',')[1])/dn for l in ls[1:]]).view(1,-1,1)
with open("2020.csv","r") as f: ls=f.readlines()
testx = torch.Tensor([float(l.split(',')[1])/dn for l in ls[:-1]]).view(1,-1,1)
testy = torch.Tensor([float(l.split(',')[1])/dn for l in ls[1:]]).view(1,-1,1)

trainx_mean = torch.mean(trainx)
trainy_mean = torch.mean(trainy)
testx_mean = torch.mean(testx)
testy_mean = torch.mean(testy)

trainx_std = torch.std(trainx)
trainy_std = torch.std(trainy)
testx_std = torch.std(testx)
testy_std = torch.std(testy)

print(trainx_mean, trainy_mean, testx_mean, testy_mean)

trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
crit = nn.MSELoss()

class Mod(nn.Module):
    def __init__(self,nhid):
        super(Mod, self).__init__()
        self.rnn = nn.RNN(1,nhid)
        self.mlp = nn.Linear(nhid,1)

    def forward(self,x):
        # x = B, T, d
        xx = x.transpose(0,1)
        y,_=self.rnn(xx)
        T,B,H = y.shape
        y = self.mlp(y.view(T*B,H))
        y = y.view(T,B,-1)
        y = y.transpose(0,1)
        return y

def test(mod):
    mod.train(False)
    totloss, nbatch = 0., 0
    for data in testloader:
        inputs, goldy = data
        haty = mod(inputs)
        loss = crit(haty,goldy)
        totloss += loss.item()
        nbatch += 1
    totloss /= float(nbatch)
    mod.train(True)
    return totloss

def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr=0.001)
    for epoch in range(nepochs):
        testloss = test(mod)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs)
            loss = crit(haty,goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        print("err",totloss,testloss)
    print("fin",totloss,testloss,file=sys.stderr)

mod=Mod(h)
print("nparms",sum(p.numel() for p in mod.parameters() if p.requires_grad),file=sys.stderr)
train(mod)
