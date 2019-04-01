from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *
from dataset import *
from pytorch_pretrained_bert.modeling import PreTrainedBertModel,BertModel


class DMBERT_Encoder(PreTrainedBertModel):
    def __init__(self,config):
        super(DMBERT_Encoder,self).__init__(config)
        self.bert=BertModel(config)
        self.dropout=nn.Dropout(p=keepProb)
        #self.M=nn.Linear(EncodedDim,dimE)
        self.maxpooling=nn.MaxPool1d(SenLen)
    def forward(self,inp,inMask,maskL,maskR):
        SZ=inp.size(0)
        conved,_=self.bert(inp,None,inMask,False)
        conved=conved.transpose(1,2)
        conved=conved.transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+torch.ones_like(L)
        R=R+torch.ones_like(R)
        pooledL=self.maxpooling(L).contiguous().view(SZ,dimC)
        pooledR=self.maxpooling(R).contiguous().view(SZ,dimC)
        pooled=torch.cat((pooledL,pooledR),1)
        pooled=pooled-torch.ones_like(pooled)
        #rep=torch.cat((pooled,loc_embeds),1)
        #rep=F.tanh(self.dropout(rep))
        return pooled
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.encoder=DMBERT_Encoder.from_pretrained("../../BERT_CACHE/bert-base-uncased")
        self.C=nn.Linear(EncodedDim,1)
    def forward(self,inp,inMask,maskL,maskR):
        reps=self.encoder(inp,inMask,maskL,maskR)
        return self.C(reps).view(inp.size(0))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.encoder=DMBERT_Encoder.from_pretrained("../../BERT_CACHE/bert-base-uncased")
        self.M=nn.Linear(EncodedDim,dimE)
    def forward(self,inp,inMask,maskL,maskR):
        reps=self.encoder(inp,inMask,maskL,maskR)
        return F.softmax(self.M(reps),dim=1)

        
