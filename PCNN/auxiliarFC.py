import torch
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torch import nn


class BatchSampler(BatchSampler):#adapted from https://github.com/adambielski/siamese-triplet

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes=self.labels_set.copy()
            print(classes)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
    

def ssim(fmc,comb,window_size=3,sigma=1.5,channels=1):

    with torch.no_grad():

      similitude=torch.zeros(comb.size()[0])
      gauss =  torch.FloatTensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
      gauss=gauss/gauss.sum()
      w1d = gauss.unsqueeze(1)
      window = w1d.mm(w1d.t()).unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor)
      pad = window_size // 2
      ant=0

      for i in range(fmc.size()[0]-1):

        img1=fmc[i].unsqueeze(0).clone()

        img2=fmc[i+1:].clone()
        
        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2 
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

        C1 = (0.01 ) ** 2
        C2 = (0.03 ) ** 2 

        numerator1 = 2 * mu12 + C1  
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1 
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
        ssim_score=abs(ssim_score.view(ssim_score.size()[0], -1).mean(1))
        val=ssim_score.size()[0]+ant
        similitude[ant:val]=ssim_score.clone()
        ant+=ssim_score.size()[0]

    return similitude

def Cos_sim(fmc,comb):
    cos= nn.CosineSimilarity(dim=0, eps=1e-6)
    similitude=torch.zeros(comb.size()[0])

    with torch.no_grad():
      ant=0     
      for i in range(fmc.size()[1]-1):
        x=fmc[:,i].unsqueeze(1).clone()
        y=fmc[:,i+1:].clone()

        score=abs(cos(x,y))
        sumx=torch.sum(x,0)

        if sumx==0:
          a=torch.where(torch.sum(y,0)==0)
          b=torch.where(score==0)
          positions= np.intersect1d(a[0].cpu(), b[0].cpu())
          score[positions]=1
        val=score.size()[0]+ant
        similitude[ant:val]=score.clone()
        ant+=score.size()[0]

    return similitude

