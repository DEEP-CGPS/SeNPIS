import torch.nn.utils.prune as prune
import torch
from . import auxiliarFC
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import operator
ssim=auxiliarFC.ssim
Cos_sim=auxiliarFC.Cos_sim
BatchSampler=auxiliarFC.BatchSampler

class Hook():
  def __init__(self, module):
    self.hook = module.register_forward_hook(self.hook_fn)
  def hook_fn(self, module, input, output):
    self.output = output.detach()
  def close(self):
    self.hook.remove()

    
class MaskSimilitudev1(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    def __init__(self,listpruning):
        self.listpruning=listpruning
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask[self.listpruning]=0
        return mask
    

class SeNPIS():
    
  def __init__(self,Net,amount,dataset,n_classes,n_samples,sigma, attenuation_coefficient,criterion):
      self.Net=Net
      self.amount=amount
      self.dataset=dataset
      self.n_classes=n_classes
      self.n_samples=n_samples
      self.sigma=sigma
      self.attenuation_coefficient=attenuation_coefficient
      self.criterion=criterion
      self.MakeBatch()
      self.run()

  def run(self):
    #dictionary of values necessary for plots
    self.dp={"list_IM_Local":[],"list_IM_LocalA":[],
             "list_IM_Global":[],"layers_names":[],"list_fm":[]}
    ActualConv=1
    ActualFc=1

    with torch.no_grad():
      L=0
      for name, module in self.Net.named_modules():
          if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
              L+=1
              
      cont=0
      for name, module in self.Net.named_modules():
        if  isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) and cont<L-1:
            if isinstance(module,torch.nn.Conv2d):
              self.val=1
              a="Conv"+str(ActualConv)
              ActualConv+=1
              
            else:
              self.val=0
              a="Fc"+str(ActualFc)
              ActualFc+=1

            self.dp["layers_names"].append(a)
            self.module=module
            cont+=1
            self.MakeHook()
            print(self.fm.size())
            self.IM()
            numpf=round(self.amount*self.fm.size()[1])
            sorted, indices = torch.sort(self.IM_Global)
            self.listpruning=indices[0:numpf]
            MaskSimilitudev1.apply(self.module, name='weight',listpruning=self.listpruning)
            prune.remove(self.module, 'weight')
            MaskSimilitudev1.apply(self.module, name='bias',listpruning=self.listpruning)
            prune.remove(self.module, 'bias')
            print(self.dp["layers_names"])

  def IM(self):
    with torch.no_grad():
      self.IM_Local=torch.empty((self.n_classes,self.fm.size()[1]))

      for n in range(self.fm.size()[1]):      
        self.weight_backup=self.module.weight[n].clone()
        self.bias_backup=self.module.bias[n].clone()
        self.module.weight[n]*=0
        self.module.bias[n]*=0  
        self.actual_loss=self.ClassLoss()
        self.IM_Local[:,n]=abs(self.actual_loss-self.loss)
        self.module.weight[n]=self.weight_backup.clone()
        self.module.bias[n]=self.bias_backup.clone()

      self.dp["list_IM_Local"].append(self.IM_Local.clone())
      self.IM_Global=self.IM_Local.clone()

      for i in range(self.n_classes):

        self.class_fm=self.fm[i*self.n_samples:i*self.n_samples+self.n_samples,:].clone()
        self.ListSimilitude(self.class_fm.clone())
        self.threshold=torch.mean(self.similitude)+(torch.std(self.similitude)*self.sigma)

        if len(torch.where(self.similitude>self.threshold)[0]) !=0:
          self.similitude[self.similitude>self.threshold]=1
          self.similitude[self.similitude<=self.threshold]=0
          self.similitude=self.similitude.squeeze(0).type(torch.BoolTensor)
          d=torch.combinations(self.IM_Global[i,:])
          t=torch.min(d, 1)
          h=self.comb[self.similitude].clone()
          ind=t.indices[self.similitude]
          h=h[torch.tensor(range(h.size()[0])),ind]
          self.IM_Global[i,:]=self.IM_Global[i,:].mul(torch.pow(self.attenuation_coefficient,torch.bincount(h,minlength=self.IM_Global.size()[1])))
      
      self.dp["list_IM_LocalA"].append(self.IM_Global.clone())
      self.IM_Global=torch.mean(self.IM_Global,dim=0)
      self.dp["list_IM_Global"].append(self.IM_Global.clone())

  def ListSimilitude(self,fmc):
    with torch.no_grad():
      if self.val==1:
        self.fmc=torch.mean(fmc,0).unsqueeze(1)
        self.comb = torch.combinations(torch.tensor(range(self.fmc.size()[0])))
        self.similitude= ssim(self.fmc.clone(),self.comb)
      else:
        self.fmc=fmc
        self.comb = torch.combinations(torch.tensor(range(self.fmc.size()[1])))
        self.similitude= Cos_sim(self.fmc.clone(),self.comb)          


  def ClassLoss(self):
    with torch.no_grad():
      loss=torch.empty(self.n_classes)
      output=self.Net(self.x)
      for i in range(self.n_classes):
        loss[i]=self.criterion(output[i*self.n_samples:i*self.n_samples
                              +self.n_samples,:].to(self.device),
                        self.labels[i*self.n_samples:i*self.n_samples
                              +self.n_samples].to(self.device)).clone().detach()     
      return loss

  def MakeBatch(self):
      
      self.device=next(self.Net.parameters()).device
      
      if isinstance(self.dataset[0], str):
        self.seed=self.dataset[1]
        if self.dataset[0]=="Moons":
          from sklearn.datasets import make_moons
          x,y = make_moons(n_samples=2000,shuffle=False, noise=0.08,random_state=self.seed)
        elif self.dataset[0]=="Circles":
          from sklearn.datasets import make_circles
          x,y = make_circles(n_samples=2000,shuffle=False, 
                                  noise=0.1,random_state=self.seed,factor=.5)
        elif self.dataset[0]=="Blobs":
          from sklearn.datasets import make_blobs
          x,y=make_blobs(n_samples=3000,
                             cluster_std=[1.1, 1.2, 1.3],random_state=self.seed)
          
        else:
          raise ValueError('Incorrect dataset name, the available datasets by name are: Moons, Circles and Blobs')
        
        x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
        
        
        _,x_val,_,y_val=train_test_split(x_train, y_train, 
                                         test_size=round((self.n_samples)/len(y),3)*self.n_classes, 
                                         stratify=y_train)
        print(x_val)
        x_g,y_g = zip(*sorted(list(zip(x_val, y_val)), key = operator.itemgetter(1)))
        
        self.x=torch.tensor(list(x_g)).type(torch.float32).to(self.device)
        self.labels=torch.tensor(list(y_g)).type(torch.int64).to(self.device)

      else:
        bbatch_sampler = BatchSampler(self.dataset,
                                      self.n_classes, self.n_samples)
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_sampler=bbatch_sampler)
        self.meh=dataloader
        batchp= iter(dataloader)
        self.x, self.labels = batchp.next()

      
      self.x=self.x.to(self.device)
      
  def MakeHook(self):
      h1=Hook(self.module)
      _=self.Net(self.x)
      h1.close()
      self.fm=F.relu(h1.output)
      self.dp["list_fm"].append(self.fm.clone())
      self.loss=self.ClassLoss()