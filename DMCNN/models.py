from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *
from dataset import *

class DMCNN_Encoder(nn.Module):
    def __init__(self):
        super(DMCNN_Encoder,self).__init__()
        self.word_emb=nn.Embedding(len(wordVec),dimWE,padding_idx=0)
        #self.word_emb=nn.Embedding.from_pretrained(torch.FloatTensor(wordVec),freeze=False,padding_idx=0)
        weight=torch.tensor(wordVec)
        weight.requires_grad_(True)
        self.word_emb.weight.data.copy_(weight)
        #self.word_emb.weight.requires_grad_(True)
        #print(self.word_emb.weight.data[0])
        self.pos_emb=nn.Embedding(MaxPos,dimPE)
        self.conv=nn.Conv1d(dimWE+dimPE,dimC,filter_size,padding=1)
        self.dropout=nn.Dropout(p=keepProb)
        #self.M=nn.Linear(EncodedDim,dimE)
        self.maxpooling=nn.MaxPool1d(SenLen)
    def forward(self,inp,pos,loc,maskL,maskR):
        SZ=inp.size(0)
        embeds=self.word_emb(inp)
        pos_embeds=self.pos_emb(pos)
        loc_embeds=self.word_emb(loc).contiguous().view(SZ,(2*LocalLen+1)*dimWE)
        #print("loc",loc_embeds.size())
        wordVec=torch.cat((embeds,pos_embeds),2).transpose(1,2)
#        print(wordVec.size())
        conved=self.conv(wordVec).transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+torch.ones_like(L)
        R=R+torch.ones_like(R)
        #print("Left",L)
        #print("Right",R)
        pooledL=self.maxpooling(L).contiguous().view(SZ,dimC)
        pooledR=self.maxpooling(R).contiguous().view(SZ,dimC)
        #print("Left",pooledL)
        #print("Right",pooledR)
        pooled=torch.cat((pooledL,pooledR),1)
        pooled=pooled-torch.ones_like(pooled)
        #print("Pooled",pooled)
        #rep=loc_embeds
        rep=torch.cat((pooled,loc_embeds),1)
        rep=F.tanh(self.dropout(rep))
        return rep
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.encoder=DMCNN_Encoder()
        self.C=nn.Linear(EncodedDim,1)
    def forward(self,inp,pos,loc,maskL,maskR):
        reps=self.encoder(inp,pos,loc,maskL,maskR)
        return self.C(reps).view(inp.size(0))
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.encoder=DMCNN_Encoder()
        self.M=nn.Linear(EncodedDim,dimE)
    def forward(self,inp,pos,loc,maskL,maskR):
        reps=self.encoder(inp,pos,loc,maskL,maskR)
        return F.softmax(self.M(reps),dim=1)
        #return self.M(reps)
        
