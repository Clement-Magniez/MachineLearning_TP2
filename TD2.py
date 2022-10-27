from CNN import ModCnn
from RNN import ModRnn
import torch


# modCnn=ModCnn(nhid=10, kernel_size=3, timeframe=10)
# testloader, trainloader = modCnn.load_data()
# modCnn.train_(trainloader, testloader, n_epochs=10)
# print("Done training CNN\n\n")

modRnn=ModRnn(nhid=10)
testloader, trainloader = modRnn.load_data()
modRnn.train_(trainloader, testloader, n_epochs=100)
print("Done training RNN\n\n")

for X,Y in testloader:
    x, y = X, Y

print(x.shape)
for i in range(1,6):
    pred = modRnn(x[:,:i*int(360/6)]) * torch.sqrt(modRnn.testx_std) + modRnn.testx_mean
    ground_truth = y[0][i*int(360/6)-1].item() * torch.sqrt(modRnn.testx_std) + modRnn.testx_mean
    print(f"RNN predicted {pred[0][-1].item():.1f}, ground truth was {ground_truth}")





modCnn=ModCnn(nhid=10, kernel_size=3, timeframe=10)
testloader, trainloader = modCnn.load_data()
modCnn.train_(trainloader, testloader, n_epochs=10)
print("Done training CNN\n\n")
i = 0
for x,y in testloader:
    pred = modCnn(x) * torch.sqrt(modCnn.testx_std) + modCnn.testx_mean
    ground_truth = y.item() * torch.sqrt(modCnn.testx_std) + modCnn.testx_mean
    print(f"CNN predicted {pred.item():.1f}, ground truth was {ground_truth}")
    i+=1
    if i ==5: break