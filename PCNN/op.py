import torch
from math import floor
import pandas as pd
import os
from datetime import datetime

class Fit():
    def __init__(self,Net=None,trainloader=None,testloader=None
                 ,valloader=None,device=None,NV=None, 
                 criterion=None, optimizer=None,Epoch=None,SaveDir="Default",Net_Name=None):
        self.Net=Net
        self.trainloader=trainloader
        self.valloader=valloader
        self.testloader=testloader
        self.device=device
        self.NV=NV
        self.criterion=criterion
        self.optimizer=optimizer
        self.Epoch=Epoch
        self.SaveDir=SaveDir
        self.Net_Name=Net_Name
    
    
    def DoEpoch(self,Net=None,trainloader=None,testloader=None
                 ,valloader=None,device=None,NV=None, 
                 criterion=None, optimizer=None,NumEpoch=None,SaveDir=None,Net_Name=None):
        
        plist=[Net,trainloader,testloader, valloader, 
               device, NV, criterion, optimizer,NumEpoch,SaveDir,Net_Name]
        pliststr=["Net","trainloader","testloader","valloader", 
               "device", "NV", "criterion", "optimizer","NumEpoch","SaveDir","Net_Name"]

        if not os.path.exists(self.SaveDir):
            os.makedirs(self.SaveDir)
            
        for i,j in zip(plist,pliststr):
            if i!=None:
                setattr(self, j,i)
                
        if(self.device==None):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        val=floor(len(self.trainloader.dataset)/(self.trainloader.batch_size*self.NV))
        
        if self.valloader!=None:
            valv=floor(len(self.valloader.dataset)/(self.trainloader.batch_size))
        
        if self.NumEpoch==0 and self.valloader!=None:
            print('      ---------------------------------------------------------------')
            print('      --------------------------Training Start-----------------------')
            print('      ---------------------------------------------------------------')
            data = {'    Train Loss   ':[''], '    Val Loss  ':[''], '   Train ACC   ':[''], '   VAL ACC   ':['']} 
            df = pd.DataFrame(data, index = ['']) 
            print(df)
            self.PastAcc=0
        elif self.NumEpoch==0:
            print('      ---------------------------------------------------------------')
            print('      --------------------------Training Start-----------------------')
            print('      ---------------------------------------------------------------')
            data = {'    Train Loss   ':[''], '    Train Accuracy   ':['']} 
            df = pd.DataFrame(data, index = ['']) 
            print(df)
            self.PastAcc=0
        # else:
        #     self.PastAcc=self.ActualAcc
        running_loss = 0.0
        correcttrain=0.0
        totaltrain=0.0
        for i, data in enumerate(self.trainloader, 0):

            inputs, labels = data[0].to(self.device),data[1].to(self.device)

            self.optimizer.zero_grad()
    
            outputs = self.Net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
    
            running_loss += loss.item()
            totaltrain += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correcttrain += (predicted == labels).sum().item()
            
            if i % val == val-1 and self.valloader!=None:
                totalval=0
                correctval=0
                running_lossval=0
                with torch.no_grad():
                    for data in self.valloader:
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        outputs= self.Net(inputs)
                        loss = self.criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        totalval += labels.size(0)
                        correctval += (predicted == labels).sum().item()
                        running_lossval += loss.item()
                self.ActualAcc=correctval/totalval
                if self.ActualAcc>self.PastAcc:
                    self.save()
                    self.PastAcc=self.ActualAcc
                
                data = {'             ':[running_loss/val], '              ':[running_lossval/valv], 
                        '                 ':[correcttrain/totaltrain], '               ':[correctval/totalval]} 
                df = pd.DataFrame(data, index = [self.NumEpoch+1]) 
                print(df)
                running_loss = 0.0
                totaltrain=0
                correcttrain=0
            elif i % val == val-1 and self.valloader==None:
                
                self.ActualAcc=correcttrain/totaltrain
                if self.ActualAcc>self.PastAcc:
                    self.save()
                    self.PastAcc=self.ActualAcc
                
                data = {'             ':[running_loss/val],'                 ':[correcttrain/totaltrain]}
                df = pd.DataFrame(data, index = [self.NumEpoch+1]) 
                print(df)
                running_loss = 0.0
                totaltrain=0
                correcttrain=0
                
    def DoTrain(self,Net=None,trainloader=None,testloader=None
                 ,valloader=None,device=None,NV=None, 
                 criterion=None, optimizer=None, Epoch=None,SaveDir=None,Net_Name=None):
        
        plist=[Net,trainloader,testloader, valloader, 
               device, NV, criterion, optimizer,Epoch,SaveDir,Net_Name]
        pliststr=["Net","trainloader","testloader","valloader", 
               "device", "NV", "criterion", "optimizer","Epoch","SaveDir","Net_Name"]
        for i,j in zip(plist,pliststr):
            if i!=None:
                setattr(self, j,i)
                
        for epoch in range(self.Epoch):
            self.DoEpoch(NumEpoch=epoch)

            
    def save(self):
        now = datetime.now()
        print("Your network has been saved!!", now)
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        
        if self.Net_Name==None: 
            name="ACC-"+str(float("{:.2f}".format(self.ActualAcc*100)))+"-Time-"+'-'+dt_string
        else:
            name=self.Net_Name
        
        torch.save(self.Net,self.SaveDir+'/'+name+'.pth')
    

        