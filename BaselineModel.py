import torch
import torch.nn as nn
import torchvision
import einops
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import os
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
!pip install torchmetrics
from torchmetrics import F1Score

batch_size = 128
device = 'cuda'
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
    def __init__(self,embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(self.embed_size,self.embed_size)
        self.key = nn.Linear(self.embed_size, self.embed_size)
        self.value = nn.Linear(self.embed_size,self.embed_size)
        self.softmax = nn.Softmax(dim =1)
    def forward(self,embed: torch.Tensor):
        query = self.query(embed)
        key  = einops.rearrange(self.key(embed),"b n e ->b e n")
        value = self.value(embed)
        sdp = torch.matmul(self.softmax(torch.matmul(query,key)/key.size(dim = 100)**(1/2)),value)
        return sdp

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.embed_part = int(self.embed_size/self.num_heads)
        self.DotProductAttention = DotProductAttention(self.embed_part)
        
    def forward(self,embed):
        
        splitted_embed = torch.tensor_split(embed,self.num_heads,dim = 100)
        spds = [self.DotProductAttention(i) for i in splitted_embed]
        return torch.concat(spds,dim = 100)

class LinearProjection(nn.Module):
    def __init__(self,N:int,in_channels,embed_size,batch_size):
        super().__init__()
        self.N = N #patch size
        self.batch_size = batch_size 
        self.embed_size = embed_size # The embedding size is the size of the patch of the image after it has been linearly projected.
        self.in_channels = in_channels # The number of channels in the image, where 1 indicates a grayscale image and 3 indicates a colored image.
        self.linear = nn.Linear(self.in_channels*self.N**2,self.embed_size) # Linear projection layer
        self.class_token = nn.Parameter(torch.randn(self.batch_size,1,embed_size))
        self.position_embed = nn.Parameter(torch.randn(self.N**2+1,self.embed_size))
    # Reshape the input tensor from having dimensions (number of batches, number of channels, width, height) to (number of batches, number of patches, embedding size)
    def forward(self,x:torch.Tensor):
        out = einops.rearrange(x,"b c (h px) (w py) ->b (h w) (c px py)",px =self.N,py =self.N)
        out = self.linear(out)
        
        out = torch.cat([out,self.class_token],dim =1)
        out = out+self.position_embed
        return out

class EncoderBlock(nn.Module):
    def __init__(self,embed_size,num_heads,N,in_channels,batch_size):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.N = N
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.mha = MultiHeadAttention(embed_size,num_heads)
        self.Linear = nn.Linear(embed_size,embed_size)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.proj = LinearProjection(self.N,self.in_channels,self.embed_size,self.batch_size)
    def forward(self,embed):
        out = self.mha(embed)
        out = out + embed
        out = self.layernorm(out)
        out = self.Linear(out)
        return out

class Transformer(nn.Module):
    def __init__(self,embed_size,num_heads,N,in_channels,batch_size,num_encoders,num_class,device):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.N = N
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.num_encoders = num_encoders
        self.num_class = num_class
        self.proj = LinearProjection(self.N,self.in_channels,self.embed_size,self.batch_size)

        self.tiny_block = [nn.Sequential(nn.LayerNorm(self.embed_size),EncoderBlock(self.embed_size,self.num_heads,self.N,self.in_channels,self.batch_size)) for i in range(num_encoders)]
        self.block_seq = nn.Sequential(*self.tiny_block)
        
        self.Linear = nn.Linear(self.embed_size,self.num_class)
        
        self.layernorm = nn.LayerNorm(num_class)

        indices = [i for i in range(self.num_class)]
        self.indices = torch.tensor(indices)
    
    def num_of_parameters(self,):

        return sum(p.numel() for p in self.parameters())
    
    def forward(self,img):
        out = self.proj(img)
        out = self.block_seq(out)
        out = self.Linear(torch.squeeze(torch.index_select(out,1,torch.tensor(self.N**2).to(self.device))))
        out = self.layernorm(out)
        return out

f1 = F1Score(num_classes=100).to(device)

def train(trainloader, validloader, model, optimizer, criterion,epochs,f1,batch_size):
    device = "cuda"
    valid_loss_min = np.Inf
    for i in range(epochs):
        print("Epoch - {} Started".format(i+1))

        train_loss = 0.0
        valid_loss = 0.0
        train_score = 0.0
        val_score = 0.0
        model.train()
        for data, target in tqdm(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            train_score = train_score +f1(output,target)
        model.eval()
        for data, target in validloader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
            val_score = val_score + f1(output,target)
        train_loss = train_loss/len(trainloader.sampler)
        valid_loss = valid_loss/len(validloader.sampler)
        train_score = batch_size*train_score/len(trainloader.sampler)
        val_score = batch_size*val_score/len(validloader.sampler)
        print(f"F1 Score for train: {train_score}, F1 Score for validation: {val_score} ")
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            i+1, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model'.format(
                valid_loss_min,
                valid_loss))
            
            torch.save(model.state_dict(), 'baselinemodel.pt')
            valid_loss_min = valid_loss

model = Transformer(embed_size = 512,num_heads = 8,N = 8,in_channels =3,batch_size = 128,num_encoders = 8,num_class = 100,device = "cuda").to(device)
beta1 = 0.9
beta2 = 0.990
weight_decay = 0.3
learning_rate = 0.000125
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate,betas = (beta1,beta2),weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
train(trainloader,testloader,model,optimizer,criterion,100,f1,batch_size)