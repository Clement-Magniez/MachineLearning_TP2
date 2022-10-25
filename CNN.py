import sys
import torch.nn as nn
import torch

h=10
nepochs=20
timeframe=10
kernel_size=3

with open("2019.csv","r") as f: ls=f.readlines()
basetrainx = torch.Tensor([float(l.split(',')[1]) for l in ls]).view(1,-1,1)
with open("2020.csv","r") as f: ls=f.readlines()
basetestx = torch.Tensor([float(l.split(',')[1]) for l in ls]).view(1,-1,1)

basetrainx_mean = torch.mean(basetrainx)
basetestx_mean = torch.mean(basetestx)

basetrainx_std = torch.std(basetrainx)
basetestx_std = torch.std(basetestx)

basetrainx = basetrainx - basetrainx_mean / torch.sqrt(basetrainx_std)
basetestx = basetestx - basetestx_mean / torch.sqrt(basetestx_std)

trainx, trainy, testx, testy = [], [], [], []
for i in range(365-timeframe-1):
    trainx.append(basetrainx[0, i:i+timeframe])
    trainy.append(basetrainx[0, i+timeframe])
    testx.append(basetestx[0, i:i+timeframe])
    testy.append(basetestx[0, i+timeframe])

trainx = torch.stack(trainx)
trainy = torch.stack(trainy)
testx = torch.stack(testx)
testy = torch.stack(testy)

print(trainx.shape, trainy.shape, testx.shape, testy.shape)

print(basetrainx_mean, basetestx_mean, basetrainx_std, basetestx_std)

trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
crit = nn.MSELoss()

class ModCnn(nn.Module):
    def __init__(self, nhid, kernel_size, timeframe):
        self.nhid = nhid
        self.kernel_size = kernel_size
        self.timeframe = timeframe

        super(ModCnn, self).__init__()
        self.cnn = nn.Conv1d(1, nhid, kernel_size=kernel_size)
        self.mlp = nn.Linear(nhid*(timeframe-kernel_size+1), 1)

    def forward(self, x):
        # x = batchsize=1, timeframe=variable, nchannels=1
        xx = x.transpose(1, 2)
        # xx = N, Hin, L
        y = self.cnn(xx)
        y = self.mlp( y.reshape(1*self.nhid*(self.timeframe-self.kernel_size+1)))
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

modCnn=ModCnn(h, kernel_size, timeframe)
print("nparms",sum(p.numel() for p in modCnn.parameters() if p.requires_grad),file=sys.stderr)
train(modCnn)
print(testx[:1].shape)

print(f"{modCnn(testx[:1]) * torch.sqrt(basetestx_std) + basetestx_mean}, {testy[:1] * torch.sqrt(basetestx_std) + basetestx_mean}")