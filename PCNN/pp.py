from . import pmethods
import torch
from torch import nn

class Pruning():
    """
    ...

    Attributes
    ----------
    module : Tensor
        Tensor of specific layer or layers
    criterion : str
        name of pruning criterion to use
    name : str
        name identifies the parameter within that module using its string 
        identifier
    amoount: int or list
        % of pruning 
    n: int
        the order of norm
    dim: int
        index of the dim along which we define channels to prune

    Methods
    -------
    PruneNet():
        Applies the pruning criteria to the module, and return the pruned
        module.
    
    Reestructuring(Network):
        Remove all zero values in the network, and return a network with new
        structure
        
    """

    def __init__(self, module=None,Net=None,criterion='SeNPIS', name=None, 
                 amount=None, dataset=None,
                 n_classes=None,n_sample=None, remove=True,
                 sigma=None, 
                attenuation_coefficient=None,loss_criterion=None) -> object:
        
        self.module = module
        self.Net = Net
        self.criterion = criterion
        self.name = name
        self.amount = amount
        self.NetType='Sequential_CNN'
        self.dataset=dataset
        self.n_classes=n_classes
        self.n_sample=n_sample
        self.actual_layer=""
        self.past_layer="None"
        self.actual_index=[]
        self.past_index=[]
        self.last=False
        self.remove=remove
        self.sigma=sigma
        self.attenuation_coefficient=attenuation_coefficient
        self.loss_criterion=loss_criterion

    def PruneNet(self):
                        
        self.Sequential_CNN()
              
    def SeNPIS(self):
        
        pmethods.SeNPIS(self.Net,self.amount,self.dataset,
                              self.n_classes,self.n_sample,self.sigma,self.attenuation_coefficient,self.loss_criterion)

        
    def Reestructuring(self,module=None,last=False,actual_layer=None, 
                       past_layer=None,actual_index=None,past_index=None,past_out_size=None):
        
        plist=[module,actual_layer, past_layer,actual_index,past_index,past_out_size]
        pliststr=["module","actual_layer", "past_layer"
                  ,"actual_index","past_index","past_out_size"]
        for i,j in zip(plist,pliststr):
            if i!=None:
                setattr(self, j,i)
                
        self.last=last     
        self.device = self.module.weight.device
                
        if isinstance(self.module,torch.nn.Conv2d):
            self.actual_layer="Conv2d"
            self.past_out_size=self.module.out_channels
        elif isinstance(self.module,torch.nn.Linear):
            self.actual_layer="Linear"   
        elif isinstance(self.module,torch.nn.BatchNorm2d):
            self.actual_layer="Norm"
            
        if self.last==False and self.actual_layer!="Norm":
            self.index()
            
        if self.past_layer!="None":
            self.LayerT=self.past_layer+"to"+self.actual_layer
        else:
            self.LayerT=self.actual_layer+"to"+self.actual_layer

        getattr(self,self.LayerT)()
        
        if self.last==False:
            self.past_layer=self.actual_layer
            self.past_index=self.actual_index.clone()   
        else:
            self.past_layer="None"
            self.past_index=[]
        
    def Sequential_CNN(self):
        
        L=0
        for name, module in self.Net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                L+=1
        
        if self.criterion!="None":               
            getattr(self, self.criterion)()
        
        if self.remove==True:
            cont=0
            for name, module in self.Net.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    cont+=1
                    self.Reestructuring(module) if cont!=L else self.Reestructuring(module,last=True)      
                if isinstance(module,torch.nn.BatchNorm2d):
                    self.Reestructuring(module)
                
    def index(self):
        if self.actual_layer=='Conv2d':
            self.actual_index=self.module.weight.sum(dim=1).sum(dim=1).sum(dim=1).clone()
            self.actual_index=torch.nonzero(self.actual_index).clone()
            self.actual_index=self.actual_index.reshape(len(self.actual_index))
        elif self.actual_layer=='Linear':
            self.actual_index=self.module.weight.sum(dim=1).clone()
            self.actual_index=torch.nonzero(self.actual_index).clone()
            self.actual_index=self.actual_index.reshape(len(self.actual_index))

    
    def Conv2dtoConv2d(self):
        if self.last==True:
            self.actual_layer="EndLayer"
        if self.past_layer=="None":
            new_w=torch.index_select(self.module.weight, 0, self.actual_index).clone()
            new_b=torch.index_select(self.module.bias, 0, self.actual_index).clone() 
            new_out=new_b.size()[0]
            self.module.bias=nn.Parameter(new_b.clone())
            self.module.out_channels=new_out
        elif self.past_layer=="Conv2d" and self.actual_layer=="Conv2d":
            self.module.in_channels=self.past_index.size()[0]
            new_w=torch.index_select(self.module.weight, 1, self.past_index).clone()
            new_w=torch.index_select(new_w, 0, self.actual_index).clone()
            new_b=torch.index_select(self.module.bias, 0, self.actual_index).clone() 
            new_out=new_b.size()[0]
            self.module.bias=nn.Parameter(new_b.clone())
            self.module.out_channels=new_out
        elif self.past_layer=="Conv2d" and self.actual_layer=="EndLayer":
            self.module.in_channels=self.past_index.size()[0]
            new_w=torch.index_select(self.module.weight, 1, self.past_index).clone()
            new_b=new_out=0
            
        self.module.weight=nn.Parameter(new_w.clone())
        
    def Conv2dtoLinear(self):
        hw=int(self.module.in_features/self.past_out_size)
        real_index=[list(range(i*hw,i*hw+hw)) for i in self.past_index]
        real_index=torch.tensor(real_index)
        self.module.in_features=real_index.size()[0]*hw
        real_index=real_index.reshape(real_index.shape[0]*real_index.shape[1]).to(self.device)
        new_w=torch.index_select(self.module.weight, 1, real_index).clone()
        if self.last==False:
            new_w=torch.index_select(new_w, 0, self.actual_index).clone()       
            new_b=torch.index_select(self.module.bias, 0, self.actual_index).clone() 
            new_out=new_b.size()[0]
            self.module.bias=nn.Parameter(new_b.clone())
            self.module.out_features=new_out
        else:
            new_b=new_out=0

        self.module.weight=nn.Parameter(new_w.clone())

    
    def LineartoLinear(self):
        if self.last==True:
            self.actual_layer="EndLayer"
        if self.past_layer=="None":
            new_w=torch.index_select(self.module.weight, 0, self.actual_index).clone()       
            new_b=torch.index_select(self.module.bias, 0, self.actual_index).clone() 
            new_out=new_b.size()[0]
            self.module.bias=nn.Parameter(new_b.clone())
            self.module.out_features=new_out
        elif self.past_layer=="Linear" and self.actual_layer=="Linear":
            self.module.in_features=self.past_index.size()[0]
            new_w=torch.index_select(self.module.weight, 1, self.past_index).clone()
            new_w=torch.index_select(new_w, 0, self.actual_index).clone()
            new_b=torch.index_select(self.module.bias, 0, self.actual_index).clone() 
            new_out=new_b.size()[0]
            self.module.bias=nn.Parameter(new_b.clone())
            self.module.out_features=new_out
        elif self.past_layer=="Linear" and self.actual_layer=="EndLayer":
            self.module.in_features=self.past_index.size()[0]
            new_w=torch.index_select(self.module.weight, 1, self.past_index).clone()
            new_b=new_out=0
        self.module.weight=nn.Parameter(new_w.clone())

       
    def Conv2dtoNorm(self):
        self.module.num_features=self.past_index.size()[0]
        self.module.weight=nn.Parameter(torch.index_select(self.module.weight,0,self.past_index))
        self.module.bias=nn.Parameter(torch.index_select(self.module.bias,0,self.past_index))
        self.module.running_mean=torch.index_select(self.module.running_mean,0,self.past_index)
        self.module.running_var=torch.index_select(self.module.running_var,0,self.past_index)
        self.actual_layer=self.past_layer
        

