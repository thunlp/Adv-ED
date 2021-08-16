from __future__ import print_function
from constant import *
from models import *
from dataset import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from pytorch_pretrained_bert.optimization import BertAdam


from sklearn.metrics import f1_score,precision_score,recall_score
os.environ["CUDA_VISIBLE_DEVICES"]="2"

TestSet=Dataset("TestA")
selector=Generator().cuda()
discriminator=Discriminator().cuda()

param_optimizer=list(selector.named_parameters())
no_decay=['bias','gamma','beta']
optimizer_grouped_parameters=[
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay_rate':0.001},
        {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate':0.0}
    ]
sOpt=BertAdam(optimizer_grouped_parameters,lr=sLr,warmup=0.1,t_total=Epoch)

param_optimizer=list(discriminator.named_parameters())
no_decay=['bias','gamma','beta']
optimizer_grouped_parameters=[
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay_rate':0.001},
        {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate':0.0}
    ]
dOpt=BertAdam(optimizer_grouped_parameters,lr=dLr,warmup=0.1,t_total=Epoch)

nonNAindex=[x for x in range(1,dimE)]
bst=0.0

def test(dataset):
    discriminator.eval()
    preds=[]
    labels=[]
    for words,inMask,maskL,maskR,label in dataset.batchs():
        scores=discriminator(words.cuda(),inMask.cuda(),maskL.cuda(),maskR.cuda())
        pred=torch.argmax(scores,dim=1).cpu().numpy()
        preds.append(pred)
        labels.append(label.numpy())
    cnt=0
    cnt1=0
    FN=0
    FP=0
    preds=np.concatenate(preds,0)
    labels=np.concatenate(labels,0)
    '''
    cnt=0
    cnt1=0
    preds=np.array([1,1,1,1,0,0,0,0,0,0])
    labels=np.array([1,2,0,0,0,0,0,0,0,0])
    '''
    if dataset==TestSet or True:
        for i in range(0,preds.shape[0]):
            if labels[i]==0:
                cnt1+=1
            if preds[i]!=labels[i]:
                cnt+=1
                if preds[i]==0 and labels[i]!=0:
                    FN+=1
                if preds[i]!=0 and labels[i]==0:
                    FP+=1
        print("EVAL %s #Wrong %d #NegToPos %d #PosToNeg %d #All %d #Negs %d"%("Test",cnt,FP,FN,len(preds),cnt1))
    acc=precision_score(labels,preds,labels=list(range(1,34)),average="micro")
    f1=f1_score(labels,preds,labels=list(range(1,34)),average="micro")
    print(acc,f1)
    global bst
    if f1>bst and dataset==TestSet:
        print("BST %f"%(bst))
        torch.save(discriminator.state_dict(),"Dmodel.tar")
        bst=f1
    return f1
def genMask(idx):
    res=[]
    idx=idx.numpy()
    for x in idx:
        tmp=[0.0 for i in range(0,dimE)]
        tmp[x]=1.0
        res.append(tmp)
    return torch.ByteTensor(res)
def Dscore_G(nwords,nMask,nmaskL,nmaskR,nlabel,uwords,uMask,umaskL,umaskR,ulabel):
    nScores=F.sigmoid(discriminator(nwords.cuda(),nMask.cuda(),nmaskL.cuda(),nmaskR.cuda()))
    nScores=nScores[:,nonNAindex]
    nScores=torch.mean(nScores,dim=1)
    if uwords.size(0)==0:
        return nScores
    uScores=F.sigmoid(discriminator(uwords.cuda(),uMask.cuda(),umaskL.cuda(),umaskR.cuda()))
    umask=genMask(ulabel).cuda()
    uScores=torch.masked_select(uScores,umask)
    return torch.cat((nScores,uScores),dim=0)
def genLoss(nwords,nMask,nmaskL,nmaskR,nlabel,uwords,uMask,umaskL,umaskR,ulabel):
    dScores=Dscore_G(nwords,nMask,nmaskL,nmaskR,nlabel,uwords,uMask,umaskL,umaskR,ulabel)
    words=torch.cat((nwords,uwords),0)
    inMask=torch.cat((nMask,uMask),0)
    maskL=torch.cat((nmaskL,umaskL),0)
    maskR=torch.cat((nmaskR,umaskR),0)
    cScores=selector(words.cuda(),inMask.cuda(),maskL.cuda(),maskR.cuda())
    cScores=torch.pow(cScores,alpha)
    cScores=F.softmax(cScores,dim=0)
    return -torch.dot(cScores,torch.log(dScores))
def trainGen(unconfIter):
    sOpt.zero_grad()
    nwords,nMask,nmaskL,nmaskR,nlabel,uwords,uMask,umaskL,umaskR,ulabel=unconfIter.next()
    sLoss=genLoss(nwords,nMask,nmaskL,nmaskR,nlabel,uwords,uMask,umaskL,umaskR,ulabel)
    sLoss.backward()
    sOpt.step()
    return sLoss.item()
def disConfLoss(words,inMask,maskL,maskR,label):
    dScores=F.sigmoid(discriminator(words.cuda(),inMask.cuda(),maskL.cuda(),maskR.cuda()))
    mask=genMask(label).cuda()
    dScores=torch.masked_select(dScores,mask)
    return -torch.mean(torch.log(dScores))
def disUnconfLoss(words,inMask,maskL,maskR,label):
    cScores=selector(words.cuda(),inMask.cuda(),maskL.cuda(),maskR.cuda())
    cScores=torch.pow(cScores,alpha)
    cScores=F.softmax(cScores,dim=0)
    dScores=F.sigmoid(discriminator(words.cuda(),inMask.cuda(),maskL.cuda(),maskR.cuda()))
    mask=genMask(label).cuda()
    dScores=torch.masked_select(dScores,mask)
    return -torch.dot(cScores,torch.log(1.0-dScores))
def oneEpoch(e,joinSet,unconfSet):
    unconfIter=unconfSet.batchs()
    confIter=joinSet.conf_batch()
    cnt=0
    for uwords,uMask,umaskL,umaskR,ulabel,utimes in joinSet.unconf_batch():
        sLoss=trainGen(unconfIter)
        cwords,cMask,cmaskL,cmaskR,clabel=confIter.next()
        dLoss=disConfLoss(cwords,cMask,cmaskL,cmaskR,clabel)
        dLoss=dLoss+disUnconfLoss(uwords,uMask,umaskL,umaskR,ulabel)
        dOpt.zero_grad()
        dLoss.backward()
        dOpt.step()
        cnt+=1
        #print("Epoch %d Batch %d s_loss %f d_loss %f"%(e,cnt,sLoss,dLoss))
def testCnt(joinSet):
    discriminator.eval()
    for words,inMask,maskL,maskR,label,bound in joinSet.unconf_batch():
        scores=discriminator(words.cuda(),inMask.cuda(),maskL.cuda(),maskR.cuda())
        pred=torch.argmax(scores,dim=1).cpu().numpy()
        joinSet.utimes[bound[0]:bound[1]]+=np.equal(pred,label.numpy())
def train():
    unconfSet=uDataset("TrainA_unconf")
    joinSet=joinDataset("TrainA_conf","TrainA_unconf")
    selector.train()
    discriminator.train()
    #testCnt(joinSet)
    for e in range(0,Epoch):
        discriminator.train()
        oneEpoch(e,joinSet,unconfSet)
        testCnt(joinSet)
        test(TestSet)
    test(TestSet)
    joinSet.dump(Threshold,"TrainA_conf","TrainA_unconf")
if __name__=='__main__':
    discriminator.load_state_dict(torch.load("model.tar"))
    selector.encoder.load_state_dict(torch.load("encoder.tar"))
    test(TestSet)
    for i in range(0,ItemTimes):
        train()