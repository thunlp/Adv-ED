import os
import re
import sys
import numpy as np
from constant import *
import torch
wordVec=np.load(dataPath+"wordVec.npy")
class Dataset:
    def __init__(self,Tag):
        print(Tag)
        self.words=np.load(Tag+"_wordEmb.npy")
        self.pos=np.load(Tag+"_posEmb.npy")
        self.loc=np.load(Tag+"_local.npy")
        self.label=np.load(Tag+"_label.npy")
        self.maskL=np.load(Tag+"_maskL.npy")
        self.maskR=np.load(Tag+"_maskR.npy")
    def batchs(self):
        indices=np.random.permutation(np.arange(len(self.words)))
        self.words=self.words[indices]
        self.pos=self.pos[indices]
        self.loc=self.loc[indices]
        self.label=self.label[indices]
        self.maskL=self.maskL[indices]
        self.maskR=self.maskR[indices]
        for i in range(0,len(self.words)//BatchSize+1):
            L=i*BatchSize
            if L>=len(self.words):
                break
            R=min((i+1)*BatchSize,len(self.words))
            yield torch.LongTensor(self.words[L:R]),torch.LongTensor(self.pos[L:R]),torch.LongTensor(self.loc[L:R]),torch.FloatTensor(self.maskL[L:R]),torch.FloatTensor(self.maskR[L:R]),torch.LongTensor(self.label[L:R])
class uDataset:
    def __init__(self,Tag):
        print(Tag)
        self.words=np.load(Tag+"_wordEmb.npy")
        self.pos=np.load(Tag+"_posEmb.npy")
        self.loc=np.load(Tag+"_local.npy")
        self.label=np.load(Tag+"_label.npy")
        self.maskL=np.load(Tag+"_maskL.npy")
        self.maskR=np.load(Tag+"_maskR.npy")
    def batchs(self):
        indices=np.random.permutation(np.arange(len(self.words)))
        self.words=self.words[indices]
        self.pos=self.pos[indices]
        self.loc=self.loc[indices]
        self.label=self.label[indices]
        self.maskL=self.maskL[indices]
        self.maskR=self.maskR[indices]
        for i in range(0,len(self.words)//BatchSize+1):
            L=i*BatchSize
            if L>=len(self.words):
                break
            R=min((i+1)*BatchSize,len(self.words))
            bwords=self.words[L:R]
            bpos=self.pos[L:R]
            bloc=self.loc[L:R]
            bmaskL=self.maskL[L:R]
            bmaskR=self.maskR[L:R]
            blabel=self.label[L:R]
            nidx=np.where(blabel==0)
            uidx=np.where(blabel!=0)
            yield torch.LongTensor(bwords[nidx]),torch.LongTensor(bpos[nidx]),torch.LongTensor(bloc[nidx]),torch.FloatTensor(bmaskL[nidx]),torch.FloatTensor(bmaskR[nidx]),torch.LongTensor(blabel[nidx]),torch.LongTensor(bwords[uidx]),torch.LongTensor(bpos[uidx]),torch.LongTensor(bloc[uidx]),torch.FloatTensor(bmaskL[uidx]),torch.FloatTensor(bmaskR[uidx]),torch.LongTensor(blabel[uidx])
            
class joinDataset:
    def __init__(self,cTag,uTag):
        self.cwords=np.load(cTag+"_wordEmb.npy")
        self.cpos=np.load(cTag+"_posEmb.npy")
        self.cloc=np.load(cTag+"_local.npy")
        self.clabel=np.load(cTag+"_label.npy")
        self.cmaskL=np.load(cTag+"_maskL.npy")
        self.cmaskR=np.load(cTag+"_maskR.npy")
        self.uwords=np.load(uTag+"_wordEmb.npy")
        self.upos=np.load(uTag+"_posEmb.npy")
        self.uloc=np.load(uTag+"_local.npy")
        self.ulabel=np.load(uTag+"_label.npy")
        self.umaskL=np.load(uTag+"_maskL.npy")
        self.umaskR=np.load(uTag+"_maskR.npy")
        self.utimes=np.zeros_like(self.ulabel)
        self.it_times=len(self.uwords)//BatchSize+1
        self.c_BatchSize=len(self.cwords)//self.it_times
    def conf_batch(self):
        indices=np.random.permutation(np.arange(len(self.cwords)))
        print(self.cwords.shape)
        print(self.cpos.shape)
        print(self.cloc.shape)
        print(self.clabel.shape)
        print(self.cmaskL.shape)
        print(self.cmaskR.shape)
        self.cwords=self.cwords[indices]
        self.cpos=self.cpos[indices]
        self.cloc=self.cloc[indices]
        self.clabel=self.clabel[indices]
        self.cmaskL=self.cmaskL[indices]
        self.cmaskR=self.cmaskR[indices]
        for i in range(0,self.it_times):
            L=i*self.c_BatchSize
            if L>=len(self.cwords):
                break
            R=min((i+1)*self.c_BatchSize,len(self.cwords))
            yield torch.LongTensor(self.cwords[L:R]),torch.LongTensor(self.cpos[L:R]),torch.LongTensor(self.cloc[L:R]),torch.FloatTensor(self.cmaskL[L:R]),torch.FloatTensor(self.cmaskR[L:R]),torch.LongTensor(self.clabel[L:R])
    def unconf_batch(self):
        indices=np.random.permutation(np.arange(len(self.uwords)))
        self.uwords=self.uwords[indices]
        self.upos=self.upos[indices]
        self.uloc=self.uloc[indices]
        self.ulabel=self.ulabel[indices]
        self.umaskL=self.umaskL[indices]
        self.umaskR=self.umaskR[indices]
        self.utimes=self.utimes[indices]
        for i in range(0,self.it_times):
            L=i*BatchSize
            if L>=len(self.uwords):
                break
            R=min((i+1)*BatchSize,len(self.uwords))
            yield torch.LongTensor(self.uwords[L:R]),torch.LongTensor(self.upos[L:R]),torch.LongTensor(self.uloc[L:R]),torch.FloatTensor(self.umaskL[L:R]),torch.FloatTensor(self.umaskR[L:R]),torch.LongTensor(self.ulabel[L:R]),self.utimes[L:R]
    def dump(self,threshold,cTag,uTag):
        nc_idx=np.where(self.utimes>=threshold,1,0)
        nc_idx2=np.where(self.ulabel!=0,1,0)
        tmp=nc_idx*nc_idx2
        nc_idx=np.where(tmp==1)
        uc_idx=np.where(tmp==0)
        self.cwords=np.append(self.cwords,self.uwords[nc_idx],axis=0)
        self.cpos=np.append(self.cpos,self.upos[nc_idx],axis=0)
        self.cloc=np.append(self.cloc,self.uloc[nc_idx],axis=0)
        self.cmaskL=np.append(self.cmaskL,self.umaskL[nc_idx],axis=0)
        self.cmaskR=np.append(self.cmaskR,self.umaskR[nc_idx],axis=0)
        self.clabel=np.append(self.clabel,self.ulabel[nc_idx],axis=0)

        self.uwords=self.uwords[uc_idx]
        self.upos=self.upos[uc_idx]
        self.uloc=self.uloc[uc_idx]
        self.umaskL=self.umaskL[uc_idx]
        self.umaskR=self.umaskR[uc_idx]
        self.ulabel=self.ulabel[uc_idx]

        np.save(cTag+"_wordEmb.npy",self.cwords)
        np.save(cTag+"_posEmb.npy",self.cpos)
        np.save(cTag+"_local.npy",self.cloc)
        np.save(cTag+"_label.npy",self.clabel)
        np.save(cTag+"_maskL.npy",self.cmaskL)
        np.save(cTag+"_maskR.npy",self.cmaskR)
        
        np.save(uTag+"_wordEmb.npy",self.uwords)
        np.save(uTag+"_posEmb.npy",self.upos)
        np.save(uTag+"_local.npy",self.uloc)
        np.save(uTag+"_label.npy",self.ulabel)
        np.save(uTag+"_maskL.npy",self.umaskL)
        np.save(uTag+"_maskR.npy",self.umaskR)
