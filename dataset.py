import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def get_df2k_paths():
    eb = "/kaggle/input/datasets/anvu1204/df2kdata"
    if os.path.exists(eb):
        return (os.path.join(eb,"DF2K_train_HR"),os.path.join(eb,"DF2K_train_LR_bicubic","X4"),
                os.path.join(eb,"DF2K_valid_HR"),os.path.join(eb,"DF2K_valid_LR_bicubic","X4"))
    thr=tlr=vhr=vlr=None
    for root in ["/kaggle/input","./","../input"]:
        if not os.path.exists(root): continue
        for dp,dn,_ in os.walk(root):
            if "DF2K_train_HR" in dn: thr=os.path.join(dp,"DF2K_train_HR")
            if "DF2K_valid_HR" in dn: vhr=os.path.join(dp,"DF2K_valid_HR")
            if "DF2K_train_LR_bicubic" in dn:
                x=os.path.join(dp,"DF2K_train_LR_bicubic","X4")
                tlr=x if os.path.exists(x) else os.path.join(dp,"DF2K_train_LR_bicubic")
            if "DF2K_valid_LR_bicubic" in dn:
                x=os.path.join(dp,"DF2K_valid_LR_bicubic","X4")
                vlr=x if os.path.exists(x) else os.path.join(dp,"DF2K_valid_LR_bicubic")
    return thr,tlr,vhr,vlr

class DF2KPairedDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size_lr=64, scale=4, is_train=True, repeat=4):
        super().__init__()
        self.hr_dir,self.lr_dir,self.patch_size_lr = hr_dir,lr_dir,patch_size_lr
        self.scale,self.is_train = scale,is_train
        self.repeat = repeat if is_train else 1
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir,"*.png"))+glob.glob(os.path.join(hr_dir,"*.jpg")))
        lr_all = glob.glob(os.path.join(lr_dir,"*.png"))+glob.glob(os.path.join(lr_dir,"*.jpg"))
        lr_map = {}
        for p in lr_all:
            b=os.path.splitext(os.path.basename(p))[0]
            lr_map[b.replace(f"x{scale}","").replace(f"X{scale}","")]=p; lr_map[b]=p
        self.pairs=[]
        for hp in self.hr_files:
            b=os.path.splitext(os.path.basename(hp))[0]
            if b in lr_map: self.pairs.append((hp,lr_map[b]))
            elif f"{b}x{scale}" in lr_map: self.pairs.append((hp,lr_map[f"{b}x{scale}"]))
            
    def __len__(self): return len(self.pairs)*self.repeat
    
    def __getitem__(self, idx):
        hp,lp = self.pairs[idx%len(self.pairs)]
        hr=np.array(Image.open(hp).convert('RGB'),dtype=np.float32)/255.0
        lr=np.array(Image.open(lp).convert('RGB'),dtype=np.float32)/255.0
        if self.is_train:
            h,w,_=lr.shape; ps=self.patch_size_lr; s=self.scale
            ph,pw=max(0,ps-h),max(0,ps-w)
            if ph>0 or pw>0:
                lr=np.pad(lr,((0,ph),(0,pw),(0,0)),mode='reflect')
                hr=np.pad(hr,((0,ph*s),(0,pw*s),(0,0)),mode='reflect')
                h,w,_=lr.shape
            t,l=random.randint(0,h-ps),random.randint(0,w-ps)
            lr=lr[t:t+ps,l:l+ps,:]; hr=hr[t*s:(t+ps)*s,l*s:(l+ps)*s,:]
            if random.random()<0.5: lr=np.fliplr(lr); hr=np.fliplr(hr)
            if random.random()<0.5: lr=np.flipud(lr); hr=np.flipud(hr)
            rk=random.choice([0,1,2,3])
            if rk>0: lr=np.rot90(lr,rk); hr=np.rot90(hr,rk)
        else:
            h,w,_=lr.shape; h=(h//4)*4; w=(w//4)*4
            lr=lr[:h,:w,:]; hr=hr[:h*self.scale,:w*self.scale,:]
        return torch.from_numpy(lr.copy()).permute(2,0,1).float(), torch.from_numpy(hr.copy()).permute(2,0,1).float()
