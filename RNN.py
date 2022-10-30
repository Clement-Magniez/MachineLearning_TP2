import sys
import torch.nn as nn
import torch


class ModRnn(nn.Module):
    def __init__(self,nhid, nhid2=None):
        super(ModRnn, self).__init__()
        if nhid2 is None:
            self.rnn = nn.RNN(input_size=1,hidden_size=nhid)
        else :
            self.rnn = nn.RNN(input_size=1,hidden_size=nhid, num_layers=2)
        self.mlp = nn.Linear(nhid,1)
        self.crit = nn.MSELoss()

    def forward(self,x):
        # x = B, T, d
        xx = x.transpose(0,1)
        y,_= self.rnn(xx)
        T,B,H = y.shape
        y = self.mlp(y.view(T*B,H))
        y = y.view(T,B,-1)
        y = y.transpose(0,1)
        return y

    def train_(self, trainloader, testloader,n_epochs=50):
        optim = torch.optim.Adam(self.parameters(), lr=0.05)
        ttestl, ttrainl = [], []
        for epoch in range(n_epochs):
            testloss = self.test(testloader)
            ttestl.append(testloss)
            totloss, nbatch = 0., 0
            for data in trainloader:
                inputs, goldy = data
                optim.zero_grad()
                haty = self(inputs)
                loss = self.crit(haty,goldy)
                totloss += loss.item()
                nbatch += 1
                loss.backward()
                optim.step()
            totloss /= float(nbatch)
            ttrainl.append(totloss)
            # print("err",totloss,testloss)
        print("fin RNN",totloss,testloss,file=sys.stderr)
        return ttestl, ttrainl

    def test(self, testloader):
        self.train(False)
        totloss, nbatch = 0., 0
        for data in testloader:
            inputs, goldy = data
            haty = self(inputs)
            loss = self.crit(haty,goldy)
            totloss += loss.item()
            nbatch += 1
        totloss /= float(nbatch)
        self.train(True)
        return totloss

    def load_data(self):
        with open("2019.csv","r") as f: ls=f.readlines()
        trainx = torch.Tensor([float(l.split(',')[1]) for l in ls[:-7]]).view(1,-1,1)
        trainy = torch.Tensor([float(l.split(',')[1]) for l in ls[7:]]).view(1,-1,1)
        with open("2020.csv","r") as f: ls=f.readlines()
        testx = torch.Tensor([float(l.split(',')[1]) for l in ls[:-7]]).view(1,-1,1)
        testy = torch.Tensor([float(l.split(',')[1]) for l in ls[7:]]).view(1,-1,1)

        self.trainx_mean = torch.mean(trainx)
        self.testx_mean = torch.mean(testx)
        self.trainx_std = torch.std(trainx)
        self.testx_std = torch.std(testx)

        trainx = (trainx - self.trainx_mean) / torch.sqrt(self.trainx_std)
        testx  = (testx -  self.testx_mean ) / torch.sqrt(self.testx_std)
        trainy = (trainy - self.trainx_mean) / torch.sqrt(self.trainx_std)
        testy  = (testy -  self.testx_mean ) / torch.sqrt(self.testx_std)

        trainds = torch.utils.data.TensorDataset(trainx, trainy)
        trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
        testds = torch.utils.data.TensorDataset(testx, testy)
        testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
        return trainloader, testloader

