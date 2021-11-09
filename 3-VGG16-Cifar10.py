import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PCNN.pp import Pruning
from torchsummary import summary
from PCNN.op import Fit

# net = torchvision.models.vgg11_bn(pretrained=True)
net = torchvision.models.vgg16_bn(pretrained=True)
net.classifier[6] = nn.Linear(4096,10)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

transform = transforms.Compose(
    [transforms.Resize((224,224)),
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

print(summary(net,(3,224,224)))


#Train VGG16 with cifar10, the use of the FIT class is optional, you can use your own .fit
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
fo=Fit(Net=net,trainloader=trainloader,testloader=testloader
                  ,valloader=testloader,device=device,NV=3,
                  criterion=criterion,optimizer=optimizer,
                  SaveDir="Default",Net_Name="Best-VGG16-cifar10")
fo.DoTrain(Epoch=20)

net=torch.load("Default/Best-VGG16-cifar10.pth")
#An object of the Pruning class is created, to which the parameters needed by SeNPIS are sent
#where amount is the % of pruning, in this example 30%=0.3
Po=Pruning(Net=net,criterion='SeNPIS',
           amount=0.3,dataset=trainset,n_classes = 10, n_sample=10,sigma=1, 
                attenuation_coefficient=0.9,loss_criterion=criterion)

#The network is pruned by calling the PruneNet method
Po.PruneNet()
print(summary(net,(3,224,224)))

#The pruned net is saved "P"
torch.save(net,"Default/PrunedVGG16-cifar10-P.pth")

#The pruned network is retrained "P+FT"
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
fo=Fit(Net=net,trainloader=trainloader,testloader=testloader
                  ,valloader=testloader,device=device,NV=3,
                  criterion=criterion,optimizer=optimizer,Net_Name="PrunedVGG16-cifar10-P+FT")
fo.DoTrain(Epoch=10)
