import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pandas as pd
import seaborn as sn
from IPython.core.display import display
from torch.utils.data import random_split, DataLoader
!pip install einops
import einops
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
!pip install torchmetrics
from torchmetrics import F1Score,Accuracy
import matplotlib.pyplot as plt
import copy

#Hyperparameters
valid_per = 0.20
batch_size = 1
epochs = 20
beta1 = 0.9
beta2 = 0.990
weight_decay = 0.3
embed_size =512
num_heads = 8
patch_size = 8
in_channels = 3
num_encoders = 8
num_class = 100
relaxation_coeff = 0.03
device = "cuda"
batch_size = 128
learning_rate = 0.000125
epochs = 30
label_smooth = 0.1


# Starting with a batch size of 128
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

test_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

training_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=5)

test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=5)

class DotProductAttention(nn.Module):
    def __init__(self,relaxation_coeff,patch_size):
        super().__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.relaxation = AttnRelaxation(relaxation_coeff,patch_size)
    def forward(self,query,key,value): 
        relaxation = self.relaxation(self.softmax(torch.matmul(query,key)/key.size(dim = 2)**(1/2)))
        sdp = torch.matmul(relaxation,value)
        return sdp

class AttnRelaxation(nn.Module):
    def __init__(self,relaxation_coeff:float,patch_size:int):
        super().__init__()
        self.relaxation_coeff = relaxation_coeff
        self.T = (torch.ones((patch_size**2+1,patch_size**2+1))/patch_size).to("cuda")

    def forward(self,embed:torch.Tensor):
        return ((1-self.relaxation_coeff)*embed + self.relaxation_coeff*self.T)

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size,num_heads,relaxation_coeff,patch_size,dropout = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.DotProductAttention = DotProductAttention(relaxation_coeff,patch_size)
       
        self.key = nn.Linear(self.embed_size,self.num_heads* self.embed_size,bias = False)
        self.query = nn.Linear(self.embed_size,self.num_heads* self.embed_size,bias = False)
        self.value = nn.Linear(self.embed_size,self.num_heads* self.embed_size,bias = False)
        
        self.linear = nn.Linear(self.num_heads*self.embed_size,embed_size,bias = False)
        self.layer_norm = nn.LayerNorm(self.embed_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    def forward(self,embed):
        batch_size = embed.size(0)
        query = self.query(embed)
        key = einops.rearrange(self.query(embed),"b n e ->b e n")
        value = self.value(embed)
        sdp = self.DotProductAttention(query,key,value)
        return self.linear(self.dropout(sdp))

class EncoderBlock(nn.Module):
    def __init__(self,embed_size,num_heads,relaxation_coeff,patch_size,dropout = 0.2):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout =dropout
        self.mha = MultiHeadAttention(embed_size,num_heads,relaxation_coeff,patch_size)
        self.Linear1 = nn.Linear(self.embed_size,self.embed_size*4,bias=False)
        self.Linear2 = nn.Linear(self.embed_size*4,self.embed_size,bias = False)
        self.gelu = nn.GELU()
        self.layer_norm1 = nn.LayerNorm(self.embed_size,eps = 1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_size,eps = 1e-6)
        self.dropout = nn.Dropout(dropout)
    def forward(self,embed:torch.Tensor):
        embed = embed + self.mha(self.layer_norm1(embed))
        embed = embed+ self.Linear2(self.dropout(self.gelu(self.Linear1(self.dropout(self.layer_norm2(embed))))))
        return embed


class LinearProjection(nn.Module):
    def __init__(self,patch_size:int,in_channels:int,embed_size:int,batch_size:int):
        super().__init__()
        self.patch_size = patch_size
        self.batch_size = batch_size 
        self.embed_size = embed_size # The embedding size is the size of the patch of the image after it has been linearly projected.
        self.in_channels = in_channels # The number of channels in the image, where 1 indicates a grayscale image and 3 indicates a colored image.

        self.linear = nn.Linear(self.in_channels*self.patch_size**2,self.embed_size)
        self.class_token = nn.Parameter(torch.randn(self.batch_size,1,embed_size))
        self.position_embed = nn.Parameter(torch.randn(self.patch_size**2+1,self.embed_size))
# Reshape the input tensor from having dimensions (number of batches, number of channels, width, height) to (number of batches, number of patches, embedding size)
    def forward(self,x:int): 
        out = einops.rearrange(x,"b c (h px) (w py) -> b (h w) (c px py)",px = self.patch_size, py = self.patch_size)
        out = self.linear(out)
        out = torch.cat([out,self.class_token],dim = 1)
        out = out + self.position_embed
        return out

class ViTransformer(nn.Module):
    def __init__(self,embed_size,num_heads,patch_size,in_channels,batch_size,num_encoders,num_class,device,relaxation_coeff):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.patch_size =patch_size
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.num_encoders = num_encoders
        self.num_class = num_class
        self.proj = LinearProjection(self.patch_size,self.in_channels,self.embed_size,self.batch_size)

        self.tiny_block = [EncoderBlock(self.embed_size,self.num_heads,relaxation_coeff,patch_size) for i in range(self.num_encoders)]
        self.block_seq = nn.Sequential(*self.tiny_block)
        self.linear1 = nn.Linear(self.embed_size,self.num_class*4)
        self.linear2 = nn.Linear(self.num_class*4,self.num_class)
        self.layernorm1 = nn.LayerNorm(num_class*4)
        self.layernorm2 = nn.LayerNorm(num_class)
        self.logsoftmax = nn.LogSoftmax(dim = 0)
    def num_of_parameters(self,):

        return sum(p.numel() for p in self.parameters())
    
    def forward(self,img):
        out = self.proj(img)
        out = self.block_seq(out)
        out = self.linear1(torch.squeeze(torch.index_select(out,1,torch.tensor([self.patch_size**2]).to(self.device))))
        out = self.layernorm1(out)
        out = self.linear2(out)
        out = self.layernorm2(out)
        out = self.logsoftmax(out)
        return out

model = ViTransformer(embed_size,num_heads,patch_size,in_channels,batch_size,num_encoders,num_class,device,relaxation_coeff).to(device)

f1 = F1Score(task = 'multiclass',num_classes=num_class).to(device)

# NLL loss with label smoothing.
class LabelSmoothing_NLLL(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing_NLLL, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothing_NLLL(smoothing=label_smooth)

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate,betas = (beta1,beta2),weight_decay=weight_decay)

device = "cuda"
valid_loss_min = np.Inf
for i in range(epochs):
    print("Epoch - {}".format(i+1))
    train_loss = 0.0
    valid_loss = 0.0
    train_score = 0.0
    val_score = 0.0
    model.train()
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        train_score = train_score +f1(output,target)
    
    model.eval()
    
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        val_score = val_score + f1(output,target)

    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(testloader.sampler)
    train_score = batch_size*train_score/len(trainloader.sampler)
    val_score = batch_size*val_score/len(testloader.sampler)

    print(f"F1 Score for train: {train_score}, F1 Score for validation: {val_score} ")
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        i+1, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        best_model_wts = copy.deepcopy(model.state_dict())

        valid_loss_min = valid_loss
torch.save(best_model_wts, 'model.pt')

