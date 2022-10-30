from CNN import ModCnn
from RNN import ModRnn
import torch
import matplotlib.pyplot as plt

#### TRAINING

def train_cnn(cnn_hid, cnn_timeframe, cnn_ks):
    modCnn=ModCnn(nhid=cnn_hid, kernel_size=cnn_ks, timeframe=cnn_timeframe)
    testloader_cnn, trainloader = modCnn.load_data()
    cnn_testl, cnn_trainl = modCnn.train_(trainloader, testloader_cnn, n_epochs=10)
    return cnn_testl, cnn_trainl

def train_rnn(rnn_nhid, rnn_nhid2=None):
    modRnn=ModRnn(nhid=rnn_nhid, nhid2=rnn_nhid2)
    testloader_rnn, trainloader = modRnn.load_data()
    rnn_testl, rnn_trainl = modRnn.train_(trainloader, testloader_rnn, n_epochs=100)
    return rnn_testl, rnn_trainl

def test(modRnn, modCnn):
    testloader_rnn, _ = modRnn.load_data()
    x_rnn, y_rnn = testloader_rnn[0]
    pred_rnn = modRnn(x_rnn) * torch.sqrt(modRnn.testx_std) + modRnn.testx_mean
    ground_truth = y_rnn[0] * torch.sqrt(modRnn.testx_std) + modRnn.testx_mean

    testloader_cnn, _ = modCnn.load_data()
    pred_cnn = []
    for x,y in testloader_cnn:
        p = modCnn(x) * torch.sqrt(modCnn.testx_std) + modCnn.testx_mean
        pred_cnn.append(p.item())

    return pred_rnn, pred_cnn, ground_truth
"""
fig1, ax1 = plt.subplots(3, 3)
fig1.suptitle("CNN train dynamics")
fig2, ax2 = plt.subplots(1, 3)
fig2.suptitle("RNN train dynamics")
    
for i in range(3):
    rnn_testl, rnn_trainl = train_rnn(rnn_nhid=(i+1)*10)
    ax2[i].plot(range(len(rnn_testl)),rnn_testl,'b-',label='rnn test loss')
    ax2[i].plot(range(len(rnn_testl)),rnn_trainl,'r-',label='rnn train loss')
    ax2[i].set_title(f" {(i+1)*10} hidden neurons.")
    ax2[i].legend()
    ax2[i].set_ylim(1.5,4)
    for j in range(3):
        cnn_testl, cnn_trainl = train_cnn(cnn_hid=10*(1+i), cnn_timeframe=8, cnn_ks=2+2*j)
        
        # ax1[i, j].plot(xp[:,1],yp,'b',label='Ground truth')
        # ax1[i, j].plot(xp[:,1],pred,'r',label='Prediction')
        ax1[i, j].plot(range(len(cnn_testl)),cnn_testl,'b-',label='cnn test loss')
        ax1[i, j].plot(range(len(cnn_testl)),cnn_trainl,'r-',label='cnn train loss')
        ax1[i, j].set_title(f" : {(i+1)*10} hid., {2+2*j} k. size.")
        ax1[i, j].legend()
        ax1[i, j].set_ylim(2.5,4)
"""

fig, ax = plt.subplots(3, 3)
fig.suptitle("2 layers RNN")
for i in range(3):
    for j in range(3):
        rnn_testl, rnn_trainl = train_rnn(rnn_nhid=(i+1)*10)
        
        # ax1[i, j].plot(xp[:,1],yp,'b',label='Ground truth')
        # ax1[i, j].plot(xp[:,1],pred,'r',label='Prediction')
        ax[i, j].plot(range(len(rnn_testl)),rnn_testl,'b-',label='rnn test loss')
        ax[i, j].plot(range(len(rnn_testl)),rnn_trainl,'r-',label='rnn train loss')
        ax[i, j].set_title(f" : {(i+1)*10} on both layers, run {j+1}")
        ax[i, j].legend()
        ax[i, j].set_ylim(1.5,3.5)
plt.show()
