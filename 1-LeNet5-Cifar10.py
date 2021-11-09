import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PCNN.pp import Pruning
from torchsummary import summary
from PCNN.op import Fit
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net().to(device)

transform = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(summary(net,(3,32,32)))


#Train AlexNet with cifar10, the use of the FIT class is optional, you can use your own .fit
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
fo=Fit(Net=net,trainloader=trainloader,testloader=testloader
                  ,valloader=testloader,device=device,NV=3,
                  criterion=criterion,optimizer=optimizer,
                  SaveDir="Default",Net_Name="Best-LeNet-cifar10")
fo.DoTrain(Epoch=20)

net=torch.load("Default/Best-LeNet-cifar10.pth")
#An object of the Pruning class is created, to which the parameters needed by SeNPIS are sent
#where amount is the % of pruning, in this example 30%=0.3
Po=Pruning(Net=net,criterion='SeNPIS',
           amount=0.3,dataset=trainset,n_classes = 10, n_sample=10,sigma=1, 
                attenuation_coefficient=0.9,loss_criterion=criterion)

#The network is pruned by calling the PruneNet method
Po.PruneNet()
print(summary(net,(3,32,32)))

#The pruned net is saved "P"
torch.save(net,"Default/PrunedLeNet-cifar10-P.pth")

#The pruned network is retrained "P+FT"
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
fo=Fit(Net=net,trainloader=trainloader,testloader=testloader
                  ,valloader=testloader,device=device,NV=3,
                  criterion=criterion,optimizer=optimizer,Net_Name="PrunedLeNet-cifar10-P+FT")
fo.DoTrain(Epoch=10)
