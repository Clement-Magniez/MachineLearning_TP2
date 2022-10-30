import sys
import torch.nn as nn
import torch



class ModCnn(nn.Module):
    def __init__(self, nhid, kernel_size, timeframe):
        self.nhid = nhid
        self.kernel_size = kernel_size
        self.timeframe = timeframe

        super(ModCnn, self).__init__()
        self.cnn = nn.Conv1d(1, nhid, kernel_size=kernel_size)
        self.mlp = nn.Linear(nhid*(timeframe-kernel_size+1), 1)
        self.crit = nn.MSELoss()

    def forward(self, x):
        # x = batchsize=1, timeframe=variable, nchannels=1
        xx = x.transpose(1, 2)
        # xx = N, Hin, L
        y = self.cnn(xx)
        y = self.mlp( y.reshape(1*self.nhid*(self.timeframe-self.kernel_size+1)))
        return y

    def load_data(self):
        with open("2019.csv","r") as f: ls=f.readlines()
        trainx = torch.Tensor([float(l.split(',')[1]) for l in ls]).view(1,-1,1)
        with open("2020.csv","r") as f: ls=f.readlines()
        testx = torch.Tensor([float(l.split(',')[1]) for l in ls]).view(1,-1,1)

        self.trainx_mean = torch.mean(trainx)
        self.testx_mean = torch.mean(testx)

        self.trainx_std = torch.std(trainx)
        self.testx_std = torch.std(testx)

        trainx = (trainx - self.trainx_mean) / torch.sqrt(self.trainx_std)
        testx  = (testx -  self.testx_mean ) / torch.sqrt(self.testx_std)

        trainx_, trainy_, testx_, testy_ = [], [], [], []
        for i in range(365-self.timeframe-1-7):
            trainx_.append(trainx[0, i:i+self.timeframe])
            trainy_.append(trainx[0, i+self.timeframe+7])
            testx_.append(testx[0, i:i+self.timeframe])
            testy_.append(testx[0, i+self.timeframe+7])

        trainx = torch.stack(trainx_)
        trainy = torch.stack(trainy_)
        testx = torch.stack(testx_)
        testy = torch.stack(testy_)


        trainds = torch.utils.data.TensorDataset(trainx, trainy)
        trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=True)
        testds = torch.utils.data.TensorDataset(testx, testy)
        testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=True)

        return testloader, trainloader

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

    def train_(self, trainloader, testloader,n_epochs=20):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
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
        print("fin CNN",totloss,testloss,file=sys.stderr)
        return ttestl, ttrainl

