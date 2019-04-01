import os
import re
import sys
import numpy as np
from constant import *
import torch
#wordVec=np.load(dataPath+"wordVec.npy")

class Dataset:
    def __init__(self,Tag):
        print(Tag)
        self.words=np.load(Tag+"_wordEmb.npy")
        self.inMask=np.load(Tag+"_inMask.npy")
        self.label=np.load(Tag+"_label.npy")
        self.maskL=np.load(Tag+"_maskL.npy")
        self.maskR=np.load(Tag+"_maskR.npy")
    def batchs(self):
        indices=np.random.permutation(np.arange(len(self.words)))
        self.words=self.words[indices]
        self.inMask=self.inMask[indices]
        self.label=self.label[indices]
        self.maskL=self.maskL[indices]
        self.maskR=self.maskR[indices]
        for i in range(0,len(self.words)//BatchSize+1):
            L=i*BatchSize
            if L>=len(self.words):
                break
            R=min((i+1)*BatchSize,len(self.words))
            yield torch.LongTensor(self.words[L:R]),torch.LongTensor(self.inMask[L:R]),torch.FloatTensor(self.maskL[L:R]),torch.FloatTensor(self.maskR[L:R]),torch.LongTensor(self.label[L:R])

class uDataset:
    def __init__(self,Tag):
        print(Tag)
        self.words=np.load(Tag+"_wordEmb.npy")
        self.pos=np.load(Tag+"_inMask.npy")
        self.label=np.load(Tag+"_label.npy")
        self.maskL=np.load(Tag+"_maskL.npy")
        self.maskR=np.load(Tag+"_maskR.npy")
    def batchs(self):
        indices=np.random.permutation(np.arange(len(self.words)))
        self.words=self.words[indices]
        self.inMask=self.inMask[indices]
        self.label=self.label[indices]
        self.maskL=self.maskL[indices]
        self.maskR=self.maskR[indices]
        for i in range(0,len(self.words)//BatchSize+1):
            L=i*BatchSize
            if L>=len(self.words):
                break
            R=min((i+1)*BatchSize,len(self.words))
            bwords=self.words[L:R]
            bMask=self.inMask[L:R]
            bmaskL=self.maskL[L:R]
            bmaskR=self.maskR[L:R]
            blabel=self.label[L:R]
            nidx=np.where(blabel==0)
            uidx=np.where(blabel!=0)
            yield torch.LongTensor(bwords[nidx]),torch.LongTensor(bMask[nidx]),torch.FloatTensor(bmaskL[nidx]),torch.FloatTensor(bmaskR[nidx]),torch.LongTensor(blabel[nidx]),torch.LongTensor(bwords[uidx]),torch.LongTensor(bMask[uidx]),torch.FloatTensor(bmaskL[uidx]),torch.FloatTensor(bmaskR[uidx]),torch.LongTensor(blabel[uidx])
            
class joinDataset:
    def __init__(self,cTag,uTag):
        self.cwords=np.load(cTag+"_wordEmb.npy")
        self.cMask=np.load(cTag+"_inMask.npy")
        self.clabel=np.load(cTag+"_label.npy")
        self.cmaskL=np.load(cTag+"_maskL.npy")
        self.cmaskR=np.load(cTag+"_maskR.npy")
        #self.cindex=np.load(cTag+"_index.npy")

        #self.uindex=np.load(uTag+"_index.npy")
        '''
        for i in range(0,0):
            self.cwords=np.concatenate((self.cwords,self.cwords),axis=0)
            self.cpos=np.concatenate((self.cpos,self.cpos),axis=0)
            self.cloc=np.concatenate((self.cloc,self.cloc),axis=0)
            self.clabel=np.concatenate((self.clabel,self.clabel),axis=0)
            self.cmaskL=np.concatenate((self.cmaskL,self.cmaskL),axis=0)
            self.cmaskR=np.concatenate((self.cmaskR,self.cmaskR),axis=0)
        '''
        self.uwords=np.load(uTag+"_wordEmb.npy")
        self.uMask=np.load(uTag+"_inMask.npy")
        self.ulabel=np.load(uTag+"_label.npy")
        self.umaskL=np.load(uTag+"_maskL.npy")
        self.umaskR=np.load(uTag+"_maskR.npy")
        self.utimes=np.zeros_like(self.ulabel)
        self.it_times=len(self.uwords)//BatchSize+1
        self.c_BatchSize=len(self.cwords)//self.it_times
    def conf_batch(self):
        indices=np.random.permutation(np.arange(len(self.cwords)))
        print(self.cwords.shape)
        print(self.clabel.shape)
        print(self.cmaskL.shape)
        print(self.cmaskR.shape)
        self.cwords=self.cwords[indices]
        self.cMask=self.cMask[indices]
        self.clabel=self.clabel[indices]
        self.cmaskL=self.cmaskL[indices]
        self.cmaskR=self.cmaskR[indices]
        #self.cindex=self.cindex[indices]
        for i in range(0,self.it_times):
            L=i*self.c_BatchSize
            if L>=len(self.cwords):
                break
            R=min((i+1)*self.c_BatchSize,len(self.cwords))
            yield torch.LongTensor(self.cwords[L:R]),torch.LongTensor(self.cMask[L:R]),torch.FloatTensor(self.cmaskL[L:R]),torch.FloatTensor(self.cmaskR[L:R]),torch.LongTensor(self.clabel[L:R])
    def unconf_batch(self):
        print("MAXX %d"%(np.max(self.utimes)))
        indices=np.random.permutation(np.arange(len(self.uwords)))
        self.uwords=self.uwords[indices]
        self.uMask=self.uMask[indices]
        self.ulabel=self.ulabel[indices]
        self.umaskL=self.umaskL[indices]
        self.umaskR=self.umaskR[indices]
        self.utimes=self.utimes[indices]
        #self.uindex=self.uindex[indices]
        for i in range(0,self.it_times):
            L=i*BatchSize
            if L>=len(self.uwords):
                break
            R=min((i+1)*BatchSize,len(self.uwords))
            yield torch.LongTensor(self.uwords[L:R]),torch.LongTensor(self.uMask[L:R]),torch.FloatTensor(self.umaskL[L:R]),torch.FloatTensor(self.umaskR[L:R]),torch.LongTensor(self.ulabel[L:R]),(L,R)
    def dump(self,threshold,cTag,uTag):
        nc_idx=np.where(self.utimes>=threshold,1,0)
        nc_idx2=np.where(self.ulabel!=0,1,0)
        tmp=nc_idx*nc_idx2
        nc_idx=np.where(tmp==1)
        uc_idx=np.where(tmp==0)
        print("Transfer Size %d %d"%(nc_idx[0].shape[0],np.max(self.utimes)))
        self.cwords=np.append(self.cwords,self.uwords[nc_idx],axis=0)
        self.cMask=np.append(self.cMask,self.uMask[nc_idx],axis=0)
        self.cmaskL=np.append(self.cmaskL,self.umaskL[nc_idx],axis=0)
        self.cmaskR=np.append(self.cmaskR,self.umaskR[nc_idx],axis=0)
        self.clabel=np.append(self.clabel,self.ulabel[nc_idx],axis=0)
        #self.cindex=np.append(self.cindex,self.uindex[nc_idx],axis=0)

        self.uwords=self.uwords[uc_idx]
        self.uMask=self.uMask[uc_idx]
        self.umaskL=self.umaskL[uc_idx]
        self.umaskR=self.umaskR[uc_idx]
        self.ulabel=self.ulabel[uc_idx]
        #self.uindex=self.uindex[uc_idx]

        np.save(cTag+"_wordEmb.npy",self.cwords)
        np.save(cTag+"_inMask.npy",self.cMask)
        np.save(cTag+"_label.npy",self.clabel)
        np.save(cTag+"_maskL.npy",self.cmaskL)
        np.save(cTag+"_maskR.npy",self.cmaskR)
        #np.save(cTag+"_index.npy",self.cindex)
        
        np.save(uTag+"_wordEmb.npy",self.uwords)
        np.save(uTag+"_inMask.npy",self.uMask)
        np.save(uTag+"_label.npy",self.ulabel)
        np.save(uTag+"_maskL.npy",self.umaskL)
        np.save(uTag+"_maskR.npy",self.umaskR)
        #np.save(uTag+"_index.npy",self.uindex)
